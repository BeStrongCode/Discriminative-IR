import torch
import torch.nn as nn

class LearnableInverseDFT2d(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.H = height
        self.W = width

        # 初始化为标准 DFT 矩阵的共轭转置
        F_H = self._init_dft_matrix(height)
        F_W = self._init_dft_matrix(width)

        self.W_H_real = nn.Parameter(F_H.real.clone())
        self.W_H_imag = nn.Parameter(-F_H.imag.clone())  # 共轭
        self.W_W_real = nn.Parameter(F_W.real.clone())
        self.W_W_imag = nn.Parameter(-F_W.imag.clone())  # 共轭

    def _init_dft_matrix(self, N):
        n = torch.arange(N).float()
        k = n.view(-1, 1)
        omega = 2 * torch.pi * k * n / N  # 注意这里是正号（逆变换）
        W_real = torch.cos(omega)
        W_imag = torch.sin(omega)
        return W_real + 1j * W_imag  # (N, N)

    def forward(self, x):
        """
        x: shape (B, C, H, W), 输入频域复数张量
        返回: shape (B, C, H, W), 实值图像张量（或复数）
        """
        B, C, H, W = x.shape
        device = x.device

        W_H = torch.complex(self.W_H_real, self.W_H_imag).to(device)  # shape (H, H)
        W_W = torch.complex(self.W_W_real, self.W_W_imag).to(device)  # shape (W, W)

        x = x.to(torch.complex64)

        # 矩阵乘法：W_H * X * W_W^T
        x = torch.einsum('ij,bcjk->bcik', W_H, x)
        x = torch.einsum('ij,bcik->bcjk', W_W, x)

        # （可选）除以尺寸归一化
        return x / (H * W)

class LearnableDFT2d(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.H = height
        self.W = width

        # 实部 + 虚部参数，初始化为 DFT 矩阵或随机值
        F_H = self._init_dft_matrix(height)
        F_W = self._init_dft_matrix(width)

        # 可学习的实部和虚部
        self.W_H_real = nn.Parameter(F_H.real.clone())  # shape (H, H)
        self.W_H_imag = nn.Parameter(F_H.imag.clone())  # shape (H, H)
        self.W_W_real = nn.Parameter(F_W.real.clone())  # shape (W, W)
        self.W_W_imag = nn.Parameter(F_W.imag.clone())  # shape (W, W)

    def _init_dft_matrix(self, N):
        n = torch.arange(N).float()
        k = n.view(-1, 1)
        omega = -2 * torch.pi * k * n / N
        W_real = torch.cos(omega)
        W_imag = torch.sin(omega)
        return W_real + 1j * W_imag  # shape (N, N)

    def forward(self, x):
        """
        x: shape (B, C, H, W), 实数输入张量
        返回: shape (B, C, H, W), 复数频谱张量
        """
        B, C, H, W = x.shape
        device = x.device

        # 拼接成复数权重矩阵
        W_H = torch.complex(self.W_H_real, self.W_H_imag).to(device)  # (H, H)
        W_W = torch.complex(self.W_W_real, self.W_W_imag).to(device)  # (W, W)

        x = x.to(torch.complex64)  # (B, C, H, W)

        # 进行两次矩阵乘法：W_H * x * W_W^T
        x = torch.einsum('ij,bcjk->bcik', W_H, x)
        x = torch.einsum('ij,bcik->bcjk', W_W, x)

        return x

if __name__ == '__main__':
    H = W = 8
    x = torch.rand(1, 1, H, W).cuda()

    # 1. 初始化傅里叶和反傅里叶模块
    forward_dft = LearnableDFT2d(H, W).cuda()
    inverse_dft = LearnableInverseDFT2d(H, W).cuda()

    # 2. 执行 forward + inverse
    x_freq = forward_dft(x)
    x_rec = inverse_dft(x_freq)

    # 3. 评估重构误差
    error = torch.norm(x - x_rec.real)  # 取实部比较
    print("还原误差:", error.item())
