import cv2
import numpy as np
import os


def add_gaussian_noise(img, sigma=50):
    """
    添加随机高斯噪声
    img: uint8, [H, W, 3], range 0-255
    sigma: 噪声标准差，默认 50
    """
    img = img.astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def motion_blur_kernel(length=25, angle=0):
    """
    生成指定角度的运动模糊核，用于模拟雨条纹
    length: 雨条长度
    angle: 雨条角度，单位：度
    """
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0

    center = (length // 2, length // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel = kernel / kernel_sum

    return kernel


def add_rain_streak(
    img,
    rain_density=0.015,
    rain_length=35,
    rain_angle=-20,
    rain_intensity=0.7,
    rain_width=1,
):
    """
    添加雨条纹退化
    img: uint8, [H, W, 3], range 0-255
    rain_density: 雨滴稀疏程度，越大雨越密
    rain_length: 雨条长度
    rain_angle: 雨条角度
    rain_intensity: 雨条强度
    rain_width: 雨条宽度
    """
    h, w = img.shape[:2]

    # 生成稀疏雨点 mask
    rain_mask = np.random.rand(h, w).astype(np.float32)
    rain_mask = (rain_mask < rain_density).astype(np.float32)

    # 扩展雨条宽度
    if rain_width > 1:
        kernel_width = np.ones((rain_width, rain_width), np.uint8)
        rain_mask = cv2.dilate(rain_mask, kernel_width, iterations=1)

    # 运动模糊形成雨条
    kernel = motion_blur_kernel(length=rain_length, angle=rain_angle)
    rain_layer = cv2.filter2D(rain_mask, -1, kernel)

    rain_layer = rain_layer / (rain_layer.max() + 1e-8)
    rain_layer = rain_layer[..., None]

    img_float = img.astype(np.float32) / 255.0

    # 雨条一般偏亮
    rain_color = np.ones_like(img_float)
    rainy = img_float + rain_intensity * rain_layer * rain_color

    # 雨天整体略微变暗，可选
    rainy = rainy * 0.95

    rainy = np.clip(rainy, 0, 1)
    rainy = (rainy * 255.0).astype(np.uint8)

    return rainy


def generate_synthetic_depth(h, w, mode="vertical"):
    """
    生成伪深度图，用于大气散射模型
    depth 越大，雾越浓
    """
    if mode == "vertical":
        # 上方远、下方近：模拟自然场景远处更雾
        y = np.linspace(1.0, 0.2, h).reshape(h, 1)
        depth = np.repeat(y, w, axis=1)

    elif mode == "radial":
        # 中心近，边缘远
        yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        depth = np.sqrt(xx ** 2 + yy ** 2)
        depth = depth / (depth.max() + 1e-8)

    elif mode == "random":
        depth = np.random.rand(h, w).astype(np.float32)
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=25, sigmaY=25)
        depth = depth / (depth.max() + 1e-8)

    else:
        raise ValueError(f"Unsupported depth mode: {mode}")

    depth = depth.astype(np.float32)
    return depth


def add_haze(
    img,
    beta=1.5,
    atmospheric_light=0.85,
    depth_mode="vertical",
):
    """
    使用大气散射模型添加雾霾

    I(x) = J(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-beta * d(x))

    img: uint8, [H, W, 3], range 0-255
    beta: 散射系数，越大雾越浓
    atmospheric_light: 大气光 A，通常在 0.7-1.0
    depth_mode: vertical / radial / random
    """
    h, w = img.shape[:2]

    img_float = img.astype(np.float32) / 255.0

    depth = generate_synthetic_depth(h, w, mode=depth_mode)

    transmission = np.exp(-beta * depth).astype(np.float32)
    transmission = transmission[..., None]

    A = np.ones_like(img_float, dtype=np.float32) * atmospheric_light

    hazy = img_float * transmission + A * (1.0 - transmission)

    hazy = np.clip(hazy, 0, 1)
    hazy = (hazy * 255.0).astype(np.uint8)

    return hazy


def degrade_image(
    img,
    degradation_type="noise",
    sigma=50,
    rain_density=0.015,
    rain_length=35,
    rain_angle=-20,
    rain_intensity=0.7,
    beta=1.5,
    atmospheric_light=0.85,
    depth_mode="vertical",
):
    """
    统一退化函数

    degradation_type:
        noise: 加高斯噪声
        rain: 加雨条纹
        haze: 加雾
    """
    if degradation_type == "noise":
        return add_gaussian_noise(img, sigma=sigma)

    elif degradation_type == "rain":
        return add_rain_streak(
            img,
            rain_density=rain_density,
            rain_length=rain_length,
            rain_angle=rain_angle,
            rain_intensity=rain_intensity,
        )

    elif degradation_type == "haze":
        return add_haze(
            img,
            beta=beta,
            atmospheric_light=atmospheric_light,
            depth_mode=depth_mode,
        )

    else:
        raise ValueError(
            f"Unsupported degradation_type: {degradation_type}. "
            f"Choose from ['noise', 'rain', 'haze']."
        )


def generate_all_degradations(img):
    """
    对同一张图像生成三种退化版本
    """
    noisy = add_gaussian_noise(img, sigma=50)

    rainy = add_rain_streak(
        img,
        rain_density=0.015,
        rain_length=35,
        rain_angle=-20,
        rain_intensity=0.7,
    )

    hazy = add_haze(
        img,
        beta=1.5,
        atmospheric_light=0.85,
        depth_mode="vertical",
    )

    return {
        "noise": noisy,
        "rain": rainy,
        "haze": hazy,
    }

if __name__ == "__main__":
    img_path = "/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/clear_images/hazy/0001.png"
    save_dir = "/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/vis_results/degraded_results"
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    degraded_dict = generate_all_degradations(img)

    for name, degraded_img in degraded_dict.items():
        save_img = cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"{name}.png"), save_img)

    print("Done.")