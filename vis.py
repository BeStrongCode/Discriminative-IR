import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path):
    img = cv2.imread(path)
    assert img is not None, f"Failed to load {path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def compute_residual(deg, gt, mode="l1"):
    if mode == "l1":
        res = np.abs(deg - gt)
    elif mode == "l2":
        res = (deg - gt) ** 2
    else:
        raise ValueError
    return res.mean(axis=2)   # RGB → Gray

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def save_images(deg, gt, residual, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ 保存退化图 & GT
    cv2.imwrite(
        os.path.join(save_dir, "deg.png"),
        cv2.cvtColor((deg * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(save_dir, "gt.png"),
        cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )

    # 2️⃣ 残差灰度图
    res_gray = (residual * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, "residual_gray.png"), res_gray)

    # 3️⃣ 彩色热力图（jet）
    heatmap = plt.get_cmap("jet")(residual)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, "residual_heatmap.png"), heatmap)


base_out = "./residual_vis"

cases = {
    "dehazing": (
        "/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/input/0001_0.8_0.2.jpg",
        "/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/target/0001.png",
    ),
    "deraining": (
        "/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/train/rain-1.png",
        "/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/train/gt/norain-1.png",
    ),
    "deblurring": (
        "/media/hy/hhh/all-in-one-image-restoration/dataset/deblurring/GOPRO_Large/train1/input/000001.png",
        "/media/hy/hhh/all-in-one-image-restoration/dataset/deblurring/GOPRO_Large/train1/gt/000001.png",
    ),
    "denoising": (
        "/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/CBSD68_25/3096.png",
        "/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/CBSD68/3096.png",
    ),
    "lowlight": (
        "/media/hy/hhh/all-in-one-image-restoration/dataset/low-light/LOLdataset/train/input/2.png",
        "/media/hy/hhh/all-in-one-image-restoration/dataset/low-light/LOLdataset/train/gt/2.png",
    ),
}

for task, (deg_path, gt_path) in cases.items():
    deg = read_image(deg_path)
    gt  = read_image(gt_path)

    res = normalize(compute_residual(deg, gt))

    save_dir = os.path.join(base_out, task)
    save_images(deg, gt, res, save_dir)

    print(f"[✓] Saved {task} results to {save_dir}")
