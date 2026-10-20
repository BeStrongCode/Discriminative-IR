import torch
import torch.nn as nn
import math

def _complex_dtype_from_float(dtype: torch.dtype):
    return torch.complex64 if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128

class LearnableRFFT2(nn.Module):
    """
    可学习 rfft2：输入 (B,C,H,W) 实数，输出 (B,C,H,W//2+1) 复数
    norm: 'backward'|'ortho'|'forward' —— 语义与 torch.fft.rfft2 一致（这里只影响幅度因子）
    """
    def __init__(self, H: int, W: int, dtype=torch.float32):
        super().__init__()
        self.H = H
        self.W = W
        self.K = W // 2 + 1  # 非负频率数量（W 维压缩后长度）

        # H 维：完整 DFT (H x H)
        FH = self._init_dft_matrix(H, forward=True, dtype=dtype)  # e^{-i 2π u y/H}
        self.FH_real = nn.Parameter(FH.real.clone())  # (H,H)
        self.FH_imag = nn.Parameter(FH.imag.clone())

        # W 维：仅保留非负频率行 (K x W)
        FW_full = self._init_dft_matrix(W, forward=True, dtype=dtype)  # (W,W)
        FW_red = FW_full[:self.K, :]  # 取 k=0..W//2 行
        self.FW_red_real = nn.Parameter(FW_red.real.clone())  # (K,W)
        self.FW_red_imag = nn.Parameter(FW_red.imag.clone())

        self.norm = 'backward'

    def set_norm(self, norm: str):
        assert norm in ('backward','ortho','forward')
        self.norm = norm
        return self

    def _init_dft_matrix(self, N, forward=True, dtype=torch.float32):
        # forward=True -> e^{-i 2π k n/N}; forward=False -> e^{+i 2π k n/N}
        n = torch.arange(N, dtype=dtype)
        k = n.view(-1, 1)
        sign = -1.0 if forward else +1.0
        omega = sign * 2 * math.pi * k * n / N
        return torch.cos(omega) + 1j * torch.sin(omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W), real
        return: (B,C,H,K), complex  with K = W//2+1
        """
        assert x.dim() == 4 and x.shape[-2] == self.H and x.shape[-1] == self.W, \
            f"expected (B,C,{self.H},{self.W}), got {tuple(x.shape)}"
        B, C, H, W = x.shape
        device = x.device
        fcdtype = _complex_dtype_from_float(x.dtype)

        FH = torch.complex(self.FH_real, self.FH_imag).to(device=device, dtype=fcdtype)    # (H,H) rows: u, cols: y
        FW = torch.complex(self.FW_red_real, self.FW_red_imag).to(device=device, dtype=fcdtype)  # (K,W) rows: k, cols: x

        # 升为复数以便复权重乘法
        x_c = x.to(fcdtype)

        # 先沿 H 维做完整 DFT： (H,H) @ (B,C,H,W) -> (B,C,H,W)，H: y->u
        # u,y 对应 FH 行/列；einsum: 'uy, b c y x -> b c u x'
        X = torch.einsum('uy,bc y x->bc u x', FH, x_c)

        # 再沿 W 维做 rfft 压缩： (K,W) @ (B,C,H,W) -> (B,C,H,K)，W: x->k
        # 'kx, b c u x -> b c u k'
        X = torch.einsum('kx,bc u x->bc u k', FW, X)

        # 归一化：与 torch.fft.rfft2 保持一致（默认 backward：正变换不缩放）
        if self.norm == 'forward':
            X = X / (H * W)
        elif self.norm == 'ortho':
            X = X / math.sqrt(H * W)

        return X


class LearnableIRFFT2(nn.Module):
    """
    可学习 ifft2（全谱）：
      输入可为 complex (B, C, H, W) 或 实/虚拼接 (B, 2C, H, W)
      输出为 (B, C, H, W) 实张量（若频谱不满足厄米对称，虚部将被丢弃）
    """
    def __init__(self, H: int, W: int, dtype=torch.float32):
        super().__init__()
        self.H = H
        self.W = W

        FH = self._init_dft_matrix(H, forward=False, dtype=dtype)  # e^{+i 2π u y/H}
        FW = self._init_dft_matrix(W, forward=False, dtype=dtype)  # e^{+i 2π k x/W}

        self.FH_real = nn.Parameter(FH.real.clone())
        self.FH_imag = nn.Parameter(FH.imag.clone())
        self.FW_real = nn.Parameter(FW.real.clone())
        self.FW_imag = nn.Parameter(FW.imag.clone())

        self.norm = 'backward'  # 与 torch.fft.ifft2 默认对齐

    def set_norm(self, norm: str):
        assert norm in ('backward', 'ortho', 'forward')
        self.norm = norm
        return self

    def _init_dft_matrix(self, N, forward=True, dtype=torch.float32):
        n = torch.arange(N, dtype=dtype)
        k = n.view(-1, 1)
        sign = -1.0 if forward else +1.0
        omega = sign * 2 * math.pi * k * n / N
        return torch.cos(omega) + 1j * torch.sin(omega)

    def forward(self, X_in: torch.Tensor) -> torch.Tensor:
        """
        X_in: complex (B, C, H, W) 或 real (B, 2C, H, W) [实/虚拼接]
        return: real (B, C, H, W)
        """
        assert X_in.dim() == 4, f"expected 4D, got {tuple(X_in.shape)}"
        B, C_or_2C, H, W = X_in.shape
        assert H == self.H and W == self.W, f"Expected (H,W)=({self.H},{self.W}), got ({H},{W})"

        # # 实/虚拼接 -> 复数
        # if not torch.is_complex(X_in):
        #     assert C_or_2C % 2 == 0, "实/虚拼接通道数必须是偶数"
        #     C = C_or_2C // 2
        #     real, imag = torch.split(X_in, C, dim=1)
        #     X = torch.complex(real, imag)
        # else:
        #     X = X_in
        #     C = C_or_2C
        X = X_in
        device, dtype = X.device, X.dtype
        FH = torch.complex(self.FH_real, self.FH_imag).to(device=device, dtype=dtype)  # (H,H)
        FW = torch.complex(self.FW_real, self.FW_imag).to(device=device, dtype=dtype)  # (W,W)

        # 全谱逆变换：先 W (k->x)，再 H (u->y)
        x_tmp = torch.einsum('xk, b c u k -> b c u x', FW, X)   # (B,C,H,W)
        x     = torch.einsum('yh, b c h x -> b c y x', FH, x_tmp)

        # 归一化
        if self.norm == 'backward':
            x = x / (H * W)
        elif self.norm == 'ortho':
            x = x / math.sqrt(H * W)
        # 'forward'：不缩放

        return x.real  # 若频谱不具厄米对称，这里丢弃虚部

# B, C, H, W = 2, 3, 64, 96   # H != W
# x = torch.randn(B, C, H, W, device='cuda')
#
# rfft = LearnableRFFT2(H, W).to(x.device)
# irfft = LearnableIRFFT2(H, W).to(x.device)
#
# X = rfft(x)
# fftx = torch.cat([X.real,X.imag])
# # (B, C, 64, 49)
# x_rec = irfft(X)       # (B, C, 64, 96)