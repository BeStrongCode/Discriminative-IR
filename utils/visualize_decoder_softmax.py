#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize decoder self-attention softmax activations for Restormer-like networks.

核心功能：
1. 输入模型结构文件、权重路径、退化图像路径；
2. 自动定位 decoder_level3 / decoder_level2 / decoder_level1 / refinement 中的 Attention；
3. monkey-patch Attention.forward，抓取 attn.softmax(dim=-1)；
4. 保存每层 softmax 矩阵热力图、每个 head 热力图、注意力输出空间响应图、统计 CSV。

注意：
Restormer/MDTA 的 self-attention 是通道维注意力：
    attn shape = [B, heads, C/head, C/head]
不是 ViT 那种空间维注意力：
    [B, heads, H*W, H*W]
因此 softmax 热力图本质是“通道-通道注意力矩阵”。
脚本额外保存 attention output 的空间响应图，便于看 decoder 在图像空间上的激活位置。
"""

import os
import sys
import csv
import math
import argparse
import importlib.util
import inspect
import types
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Basic utils
# -----------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def parse_int_list(s):
    if isinstance(s, (list, tuple)):
        return list(s)
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace("\\", "_")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def import_model_class(model_file: str, model_class: str, project_root: str = None):
    if project_root:
        sys.path.insert(0, os.path.abspath(project_root))

    model_file = os.path.abspath(model_file)
    module_name = Path(model_file).stem

    spec = importlib.util.spec_from_file_location(module_name, model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, model_class):
        raise AttributeError(
            f"Cannot find class '{model_class}' in {model_file}. "
            f"Please check --model_class."
        )
    return getattr(module, model_class)


def build_model(args):
    ModelClass = import_model_class(args.model_file, args.model_class, args.project_root)

    # 这些参数与你给出的 Restormer __init__ 对齐；
    # 如果你的类不接收某个参数，下面会自动过滤。
    candidate_kwargs = dict(
        inp_channels=args.inp_channels,
        out_channels=args.out_channels,
        dim=args.dim,
        num_blocks=parse_int_list(args.num_blocks),
        num_refinement_blocks=args.num_refinement_blocks,
        heads=parse_int_list(args.heads),
        ffn_expansion_factor=args.ffn_expansion_factor,
        bias=args.bias,
        LayerNorm_type=args.LayerNorm_type,
        dual_pixel_task=args.dual_pixel_task,
    )

    sig = inspect.signature(ModelClass.__init__)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")
    model_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}

    model = ModelClass(**model_kwargs)
    return model


def strip_state_dict_prefix(state_dict):
    """Handle common checkpoint prefixes: module., model., net_g., generator."""
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dict.")

    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict

    prefixes = ["module.", "model.", "net_g.", "generator.", "params.", "params_ema."]
    new_state = {}

    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        new_state[nk] = v

    return new_state


def load_weights(model, weight_path, device, state_key=None, strict=False):
    ckpt = torch.load(weight_path, map_location=device)

    if state_key is not None:
        if state_key not in ckpt:
            raise KeyError(f"--state_key={state_key} not found in checkpoint.")
        state = ckpt[state_key]
    else:
        if isinstance(ckpt, dict):
            # 常见保存格式优先级
            possible_keys = [
                "params_ema", "params", "state_dict", "model",
                "net", "net_g", "generator", "module"
            ]
            state = None
            for k in possible_keys:
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    print(f"[INFO] Use checkpoint key: {k}")
                    break
            if state is None:
                # 有些 checkpoint 本身就是 state_dict
                state = ckpt
        else:
            state = ckpt

    state = strip_state_dict_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)

    print(f"[INFO] Load weights done. strict={strict}")
    print(f"[INFO] Missing keys: {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[WARN] First 20 missing keys:")
        for k in missing[:20]:
            print("  ", k)
    if len(unexpected) > 0:
        print("[WARN] First 20 unexpected keys:")
        for k in unexpected[:20]:
            print("  ", k)

    return model


# -----------------------------
# Image utils
# -----------------------------
def load_image_as_tensor(image_path, device, pad_multiple=8):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    h0, w0 = arr.shape[:2]

    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    pad_h = (pad_multiple - h0 % pad_multiple) % pad_multiple
    pad_w = (pad_multiple - w0 % pad_multiple) % pad_multiple

    if pad_h > 0 or pad_w > 0:
        # Restormer 下采样 3 次，通常要求 H/W 能被 8 整除
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    meta = dict(orig_h=h0, orig_w=w0, pad_h=pad_h, pad_w=pad_w)
    return x, img, meta


def tensor_to_pil(x, orig_h=None, orig_w=None):
    """
    x: [1,3,H,W] or [3,H,W], range can be arbitrary; will clamp to [0,1].
    """
    if x.dim() == 4:
        x = x[0]
    if orig_h is not None and orig_w is not None:
        x = x[:, :orig_h, :orig_w]
    x = x.detach().float().cpu().clamp(0, 1)
    arr = x.permute(1, 2, 0).numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def save_heatmap(matrix, save_path, title=None):
    matrix = np.asarray(matrix, dtype=np.float32)

    plt.figure(figsize=(6, 5), dpi=180)
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def normalize_01(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def save_gray_map(map_2d, save_path, title=None):
    map_2d = normalize_01(map_2d)

    plt.figure(figsize=(6, 6), dpi=180)
    plt.imshow(map_2d)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=8)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(input_pil, map_2d, save_path, alpha=0.45):
    """
    Overlay normalized attention response on original degraded image.
    """
    input_np = np.array(input_pil.convert("RGB")).astype(np.float32) / 255.0
    h, w = input_np.shape[:2]

    map_2d = normalize_01(map_2d)
    map_img = Image.fromarray((map_2d * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    map_np = np.array(map_img).astype(np.float32) / 255.0

    # 用 matplotlib colormap 生成伪彩色
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(map_np)[..., :3].astype(np.float32)

    overlay = (1.0 - alpha) * input_np + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)
    Image.fromarray((overlay * 255 + 0.5).astype(np.uint8)).save(save_path)


# -----------------------------
# Attention monkey-patch
# -----------------------------
def is_restormer_attention_module(module: nn.Module):
    """
    判断是否是 Restormer Attention：
    需要具备 qkv, qkv_dwconv, project_out, temperature, num_heads。
    这样不用强依赖 class name。
    """
    required_attrs = ["qkv", "qkv_dwconv", "project_out", "temperature", "num_heads"]
    return all(hasattr(module, a) for a in required_attrs)


def is_target_decoder_attention(module_name, args):
    """
    默认只抓 decoder 和 refinement。
    你也可以打开 --include_noise_level3，把 prompt 后的 noise_level3 也抓出来。
    """
    decoder_prefixes = ["decoder_level1", "decoder_level2", "decoder_level3"]
    if args.include_refinement:
        decoder_prefixes.append("refinement")
    if args.include_noise_level3:
        decoder_prefixes.append("noise_level3")

    return any(module_name.startswith(p) for p in decoder_prefixes)


def make_patched_attention_forward(module_name, collector):
    """
    替换 Attention.forward：
    原始 Restormer MDTA:
        qkv -> q,k,v -> normalize -> attn=(q@k^T)*temperature -> softmax -> attn@v -> project_out
    在 softmax 后把 attn 保存到 collector。
    """
    def patched_forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        num_heads = int(self.num_heads)
        if c % num_heads != 0:
            raise RuntimeError(
                f"Channel {c} cannot be divided by num_heads {num_heads} "
                f"at module {module_name}."
            )
        c_per_head = c // num_heads

        # [B, C, H, W] -> [B, heads, C/head, H*W]
        q = q.reshape(b, num_heads, c_per_head, h * w)
        k = k.reshape(b, num_heads, c_per_head, h * w)
        v = v.reshape(b, num_heads, c_per_head, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn_softmax = attn_logits.softmax(dim=-1)

        out = torch.matmul(attn_softmax, v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)

        # 保存 CPU 副本，避免占 GPU 显存
        collector[module_name] = {
            "attn": attn_softmax.detach().float().cpu(),  # [B, heads, C/head, C/head]
            "attn_logits": attn_logits.detach().float().cpu(),
            "attn_out": out.detach().float().cpu(),       # [B, C, H, W]
            "input_shape": (int(h), int(w), int(c)),
        }
        return out

    return patched_forward


def patch_decoder_attention(model, args):
    collector = {}
    patched_names = []

    for name, module in model.named_modules():
        if not is_target_decoder_attention(name, args):
            continue
        if not is_restormer_attention_module(module):
            continue

        module.forward = types.MethodType(make_patched_attention_forward(name, collector), module)
        patched_names.append(name)

    print(f"[INFO] Patched decoder attention modules: {len(patched_names)}")
    for n in patched_names:
        print("  ", n)

    if len(patched_names) == 0:
        print("[WARN] No target Attention module is patched.")
        print("[WARN] Please check your module names by running:")
        print("       python visualize_decoder_softmax.py --print_modules ...")

    return collector, patched_names


def print_attention_like_modules(model):
    print("[INFO] Attention-like modules:")
    count = 0
    for name, module in model.named_modules():
        if is_restormer_attention_module(module):
            print(f"  {name}  ({module.__class__.__name__})")
            count += 1
    print(f"[INFO] Total: {count}")


# -----------------------------
# Save visualization
# -----------------------------
def compute_attn_stats(attn):
    """
    attn: [B, heads, C, C]
    """
    eps = 1e-12
    entropy = -(attn * (attn + eps).log()).sum(dim=-1).mean().item()
    max_prob = attn.max(dim=-1).values.mean().item()
    min_prob = attn.min(dim=-1).values.mean().item()
    mean_prob = attn.mean().item()
    std_prob = attn.std().item()
    diag_mean = attn.diagonal(dim1=-2, dim2=-1).mean().item()
    return dict(
        entropy=entropy,
        max_prob=max_prob,
        min_prob=min_prob,
        mean_prob=mean_prob,
        std_prob=std_prob,
        diag_mean=diag_mean,
    )


def save_all_visualizations(collector, input_pil, out_dir, max_heads=8, orig_h=None, orig_w=None):
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, "decoder_attention_softmax_stats.csv")
    csv_rows = []

    for module_name, data in collector.items():
        layer_dir = os.path.join(out_dir, safe_name(module_name))
        ensure_dir(layer_dir)

        attn = data["attn"]          # [B, heads, C, C]
        attn_logits = data["attn_logits"]
        attn_out = data["attn_out"]  # [B, C, H, W]
        h, w, c = data["input_shape"]

        # 默认只看 batch=0
        attn0 = attn[0]              # [heads, C, C]
        logits0 = attn_logits[0]

        # 1) 保存平均 softmax 矩阵：mean over heads
        attn_mean = attn0.mean(dim=0).numpy()
        save_heatmap(
            attn_mean,
            os.path.join(layer_dir, "softmax_mean_over_heads.png"),
            title=f"{module_name} | softmax mean over heads | shape={attn_mean.shape}"
        )

        # 2) 保存平均 logits 矩阵：有助于判断 softmax 前是否塌缩
        logits_mean = logits0.mean(dim=0).numpy()
        save_heatmap(
            logits_mean,
            os.path.join(layer_dir, "logits_mean_over_heads.png"),
            title=f"{module_name} | logits mean over heads | shape={logits_mean.shape}"
        )

        # 3) 每个 head 单独保存
        n_heads = attn0.shape[0]
        for head_idx in range(min(n_heads, max_heads)):
            m = attn0[head_idx].numpy()
            save_heatmap(
                m,
                os.path.join(layer_dir, f"softmax_head_{head_idx:02d}.png"),
                title=f"{module_name} | head {head_idx} | softmax"
            )

        # 4) attention 输出的空间响应图
        # Restormer softmax 是通道注意力矩阵，本身不是空间图；
        # 这里用 attn_out 的通道平均绝对响应作为 decoder attention 的空间激活参考。
        spatial = attn_out[0].abs().mean(dim=0).numpy()
        if orig_h is not None and orig_w is not None:
            spatial = spatial[:orig_h, :orig_w]

        save_gray_map(
            spatial,
            os.path.join(layer_dir, "attention_output_spatial_response.png"),
            title=f"{module_name} | |attention output| mean over channels"
        )
        save_overlay(
            input_pil,
            spatial,
            os.path.join(layer_dir, "attention_output_spatial_overlay.png"),
            alpha=0.45
        )

        # 5) 统计信息
        stats = compute_attn_stats(attn)
        csv_rows.append({
            "module": module_name,
            "heads": int(attn.shape[1]),
            "matrix_size": int(attn.shape[-1]),
            "feature_h": h,
            "feature_w": w,
            "feature_c": c,
            **stats
        })

    if len(csv_rows) > 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"[INFO] Saved visualizations to: {out_dir}")
    print(f"[INFO] Saved stats CSV to: {csv_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--model_file", type=str, default="/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/net/model.py",
                        help="Path to your model architecture .py file, e.g. ./basicsr/models/archs/restormer_arch.py")
    parser.add_argument("--model_class", type=str, default="Restormer",
                        help="Model class name in --model_file.")
    parser.add_argument("--weights", type=str, default="/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/train_ckpt/3D/DisIR_2cls_AdamW10/epoch=104-step=169260.ckpt",
                        help="Path to model checkpoint.")
    parser.add_argument("--image", type=str, default="/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/images/hazy/0001_0.8_0.2.jpg",
                        help="Path to degraded input image.")
    parser.add_argument("--out_dir", type=str, default="/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/vis_results/vis_decoder",
                        help="Directory to save visualizations.")

    # Import / checkpoint
    parser.add_argument("--project_root", type=str, default=None,
                        help="Project root to add into sys.path. Useful when model_file has relative imports.")
    parser.add_argument("--state_key", type=str, default=None,
                        help="Manually specify checkpoint key, e.g. params_ema / params / state_dict.")
    parser.add_argument("--strict", type=str2bool, default=False,
                        help="Whether to strictly load checkpoint.")

    # Model args matching your Restormer
    parser.add_argument("--inp_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--num_blocks", type=str, default="4,6,6,8")
    parser.add_argument("--num_refinement_blocks", type=int, default=4)
    parser.add_argument("--heads", type=str, default="1,2,4,8")
    parser.add_argument("--ffn_expansion_factor", type=float, default=2.66)
    parser.add_argument("--bias", type=str2bool, default=False)
    parser.add_argument("--LayerNorm_type", type=str, default="WithBias")
    parser.add_argument("--dual_pixel_task", type=str2bool, default=False)

    # Visualization options
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu.")
    parser.add_argument("--pad_multiple", type=int, default=8)
    parser.add_argument("--max_heads", type=int, default=8)
    parser.add_argument("--include_refinement", type=str2bool, default=True)
    parser.add_argument("--include_noise_level3", type=str2bool, default=False,
                        help="Whether to also visualize noise_level3 attention after prompt injection.")
    parser.add_argument("--print_modules", action="store_true",
                        help="Only print all attention-like modules and exit.")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Use CPU instead.")
        args.device = "cpu"

    device = torch.device(args.device)
    ensure_dir(args.out_dir)

    model = build_model(args)
    model = load_weights(model, args.weights, device=device, state_key=args.state_key, strict=args.strict)
    model = model.to(device).eval()

    if args.print_modules:
        print_attention_like_modules(model)
        return

    collector, patched_names = patch_decoder_attention(model, args)

    x, input_pil, meta = load_image_as_tensor(args.image, device, pad_multiple=args.pad_multiple)

    # 保存输入图像副本
    input_pil.save(os.path.join(args.out_dir, "input_degraded.png"))

    with torch.no_grad():
        output = model(x)

    # 兼容你的 forward: return out_dec_level1, latent
    if isinstance(output, (tuple, list)):
        restored = output[0]
    else:
        restored = output

    restored_pil = tensor_to_pil(restored, orig_h=meta["orig_h"], orig_w=meta["orig_w"])
    restored_pil.save(os.path.join(args.out_dir, "restored_output.png"))

    if len(collector) == 0:
        print("[WARN] No attention activation collected. Please check module names.")
        print("[WARN] You can run with --print_modules to inspect attention names.")
        return

    save_all_visualizations(
        collector=collector,
        input_pil=input_pil,
        out_dir=args.out_dir,
        max_heads=args.max_heads,
        orig_h=meta["orig_h"],
        orig_w=meta["orig_w"],
    )


if __name__ == "__main__":
    main()
