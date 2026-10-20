#!/usr/bin/env python3
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from utils.dataset_utils import RealwolrdDataset, DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from train import RestormerModel
from train_5D import test_Deblur, test_Lowlight
from net.PromptIR import PromptIR
from utils import visualize_decoder_softmax as vis_softmax
from utils.visualize_decoder_softmax import str2bool


def visualize_decoder_self_attention_softmax(
    model,
    image_path,
    out_dir,
    device,
    pad_multiple=8,
    max_heads=8,
    include_refinement=True,
    include_noise_level3=False,
):
    """Visualize decoder self-attention softmax activations for one degraded image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    patch_model = model.net if hasattr(model, "net") else model
    patch_args = argparse.Namespace(
        include_refinement=include_refinement,
        include_noise_level3=include_noise_level3,
    )

    collector, patched_names = vis_softmax.patch_decoder_attention(patch_model, patch_args)
    if len(patched_names) == 0:
        raise RuntimeError(
            "No decoder attention modules were patched. "
            "Please check the model structure or set --include_refinement/--include_noise_level3 correctly."
        )

    device = torch.device(device)
    x, input_pil, meta = vis_softmax.load_image_as_tensor(image_path, device, pad_multiple=pad_multiple)
    os.makedirs(out_dir, exist_ok=True)
    input_pil.save(os.path.join(out_dir, "input_degraded.png"))

    model = model.to(device).eval()
    with torch.no_grad():
        output = model(x)

    restored = output[0] if isinstance(output, (tuple, list)) else output
    restored_pil = vis_softmax.tensor_to_pil(restored, orig_h=meta["orig_h"], orig_w=meta["orig_w"])
    restored_pil.save(os.path.join(out_dir, "restored_output.png"))

    vis_softmax.save_all_visualizations(
        collector=collector,
        input_pil=input_pil,
        out_dir=out_dir,
        max_heads=max_heads,
        orig_h=meta["orig_h"],
        orig_w=meta["orig_w"],
    )


def choose_first_decoder_attention_name(module_names):
    preferred_order = ["decoder_level1", "decoder_level2", "decoder_level3", "refinement", "noise_level3"]
    for prefix in preferred_order:
        candidates = [name for name in module_names if name.startswith(prefix)]
        if candidates:
            return sorted(candidates)[0]
    return sorted(module_names)[0] if module_names else None


def collect_decoder_layer_softmax(
    model,
    image_path,
    device,
    layer_name=None,
    pad_multiple=8,
    include_refinement=True,
    include_noise_level3=False,
):
    patch_model = model.net if hasattr(model, "net") else model
    patch_args = argparse.Namespace(
        include_refinement=include_refinement,
        include_noise_level3=include_noise_level3,
    )
    collector, patched_names = vis_softmax.patch_decoder_attention(patch_model, patch_args)
    if len(patched_names) == 0:
        raise RuntimeError(
            "No decoder attention modules were patched. "
            "Please check model structure or set include_refinement/include_noise_level3 correctly."
        )

    chosen_name = layer_name or choose_first_decoder_attention_name(patched_names)
    if chosen_name not in patched_names:
        raise ValueError(
            f"Requested decoder attention layer '{layer_name}' is not available. "
            f"Available layers: {patched_names}"
        )

    device = torch.device(device)
    x, _, _ = vis_softmax.load_image_as_tensor(image_path, device, pad_multiple=pad_multiple)
    model = model.to(device).eval()
    with torch.no_grad():
        model(x)

    attn = collector[chosen_name]["attn"]
    return attn.detach().cpu(), chosen_name


def collect_decoder_all_layers_softmax(
    model,
    image_path,
    device,
    pad_multiple=8,
    include_refinement=True,
    include_noise_level3=False,
):
    patch_model = model.net if hasattr(model, "net") else model
    patch_args = argparse.Namespace(
        include_refinement=include_refinement,
        include_noise_level3=include_noise_level3,
    )
    collector, patched_names = vis_softmax.patch_decoder_attention(patch_model, patch_args)
    if len(patched_names) == 0:
        raise RuntimeError(
            "No decoder attention modules were patched. "
            "Please check model structure or set include_refinement/include_noise_level3 correctly."
        )

    device = torch.device(device)
    x, _, _ = vis_softmax.load_image_as_tensor(image_path, device, pad_multiple=pad_multiple)
    model = model.to(device).eval()
    with torch.no_grad():
        model(x)

    attns = {
        name: collector[name]["attn"].detach().cpu()
        for name in patched_names
    }
    return attns


def compute_softmax_similarity(a, b, metric="cosine"):
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    if metric == "cosine":
        sim = F.cosine_similarity(a_flat, b_flat, dim=-1)
        return float(sim.mean().item())
    elif metric == "l2":
        dist = torch.norm(a_flat - b_flat, p=2, dim=-1)
        return float(dist.mean().item())
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


def get_image_paths_from_folder(folder_path, max_images=3):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Expected a directory for image input, got: {folder_path}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted(
        [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in exts]
    )
    if len(image_files) < max_images:
        raise ValueError(
            f"Directory must contain at least {max_images} image files, found {len(image_files)}."
        )
    return [os.path.join(folder_path, f) for f in image_files[:max_images]]


def compare_decoder_three_images_softmax_similarity(
    model,
    image_paths,
    device,
    layer_name=None,
    pad_multiple=8,
    include_refinement=True,
    include_noise_level3=False,
    metric="cosine",
):
    if len(image_paths) != 3:
        raise ValueError("Exactly three image paths are required.")

    all_attns = []
    for image_path in image_paths:
        attns = collect_decoder_all_layers_softmax(
            model,
            image_path,
            device,
            pad_multiple=pad_multiple,
            include_refinement=include_refinement,
            include_noise_level3=include_noise_level3,
        )
        all_attns.append(attns)

    layers = sorted(all_attns[0].keys())
    for attn_dict in all_attns[1:]:
        if set(attn_dict.keys()) != set(layers):
            raise RuntimeError("Decoder attention layer names do not match across images.")

    results = {}
    for layer_name in layers:
        attn0 = all_attns[0][layer_name]
        attn1 = all_attns[1][layer_name]
        attn2 = all_attns[2][layer_name]

        sim12 = compute_softmax_similarity(attn0, attn1, metric=metric)
        sim13 = compute_softmax_similarity(attn0, attn2, metric=metric)
        sim23 = compute_softmax_similarity(attn1, attn2, metric=metric)

        results[layer_name] = {
            f"{os.path.basename(image_paths[0])}__{os.path.basename(image_paths[1])}": sim12,
            f"{os.path.basename(image_paths[0])}__{os.path.basename(image_paths[2])}": sim13,
            f"{os.path.basename(image_paths[1])}__{os.path.basename(image_paths[2])}": sim23,
        }

    return {
        "metric": metric,
        "layers": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one, 4 for 5D, 5 for realworld')
    parser.add_argument('--rd_dehaze_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/WeatherBench/haze/test/", help='save path of test haze images')
    parser.add_argument('--rd_desnow_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/WeatherBench/snow/test/", help='save path of test snow images')
    parser.add_argument('--rd_derain_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/WeatherBench/rain/test/", help='save path of test rain images')
    parser.add_argument('--denoise_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/", help='save path of test hazy images')
    parser.add_argument('--deblur_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/deblurring/GOPRO_Large/test1/')
    parser.add_argument('--lowlight_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/low-light/LOLdataset/test/')
    parser.add_argument('--output_path', type=str, default="vis_results/vis_decoder", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/train_ckpt/3D/Restormer-max_pgm_gumbel_AdamW_epoch10_finetune192_2e4/epoch=3-32.82.ckpt", help='checkpoint save path')

    parser.add_argument('--vis_decoder_softmax', action='store_true', default=False,
                        help='Visualize decoder self-attention softmax activations for a single degraded image.')
    parser.add_argument('--softmax_image', type=str, default="/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/images/noise/12084.png",
                        help='Path to a degraded input image for softmax visualization.')
    parser.add_argument('--softmax_out_dir', type=str, default="vis_results/decoder_softmax_noise_pretrain",
                        help='Directory to save decoder attention softmax visualizations.')
    parser.add_argument('--softmax_max_heads', type=int, default=8,
                        help='Maximum number of heads to save individually.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run visualization on.')
    parser.add_argument('--pad_multiple', type=int, default=8,
                        help='Pad input image size to this multiple for the model.')
    parser.add_argument('--include_refinement', type=str2bool, default=True,
                        help='Whether to include refinement attention modules in visualization.')
    parser.add_argument('--include_noise_level3', type=str2bool, default=False,
                        help='Whether to include noise_level3 attention after prompt injection.')
    parser.add_argument('--compare_decoder_softmax', action='store_true', default=True,
                        help='Compare decoder all-layer softmax similarity between three images.')
    parser.add_argument('--softmax_images', type=str,  default="/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/vis_results/degraded_results",
                        help='Path to a directory containing degraded images for decoder softmax similarity. Uses the first three images alphabetically.')
    parser.add_argument('--softmax_similarity_metric', type=str, default='cosine',
                        choices=['cosine', 'l2'],
                        help='Similarity metric used for comparing softmax activations.')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA is not available; using CPU instead.')
        args.device = 'cpu'

    print(f"CKPT name : {args.ckpt_name}")

    net = RestormerModel().load_from_checkpoint(args.ckpt_name)
    # net = RestormerModel()
    # state_dict = torch.load(args.ckpt_name, map_location="cpu")
    # state_dict = state_dict['state_dict']
    # net.net.load_state_dict(state_dict, strict=True)


    net = net.to(args.device).eval()

    if args.compare_decoder_softmax:
        if args.softmax_images is None:
            raise ValueError('--softmax_images directory path is required when --compare_decoder_softmax is enabled.')
        image_paths = get_image_paths_from_folder(args.softmax_images, max_images=3)

        result = compare_decoder_three_images_softmax_similarity(
            model=net,
            image_paths=image_paths,
            device=args.device,
            layer_name=None,
            pad_multiple=args.pad_multiple,
            include_refinement=args.include_refinement,
            include_noise_level3=args.include_noise_level3,
            metric=args.softmax_similarity_metric,
        )
        print(f"[INFO] Decoder softmax similarity across all patched layers (metric={result['metric']}):")
        for layer_name, pairs in result['layers'].items():
            print(f"[LAYER] {layer_name}")
            for pair, value in pairs.items():
                print(f"  {pair}: {value:.6f}")
        return

    if args.vis_decoder_softmax:
        if args.softmax_image is None:
            raise ValueError('--softmax_image is required when --vis_decoder_softmax is enabled.')

        visualize_decoder_self_attention_softmax(
            model=net,
            image_path=args.softmax_image,
            out_dir=args.softmax_out_dir,
            device=args.device,
            pad_multiple=args.pad_multiple,
            max_heads=args.softmax_max_heads,
            include_refinement=args.include_refinement,
            include_noise_level3=args.include_noise_level3,
        )
        return

    print('[INFO] No visualization mode selected. Use --vis_decoder_softmax or --compare_decoder_softmax.')


if __name__ == '__main__':
    main()
    
