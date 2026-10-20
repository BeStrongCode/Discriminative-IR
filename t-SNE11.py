import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from train import RestormerModel

# --------------------- 基础设置 ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
ckpt_path = 'train_ckpt/5D/Restormer-max_pgm_gumbel_AdamW10_fintune192/epoch=116-30.45.ckpt'
dataset_path = "5D-images"
save_path = 't-SNE/DRTPIR_5D_latent_framed.png'
outlier_save_txt = 't-SNE/outliers-5D.txt'

# --------------------- 数据加载 ---------------------
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# --------------------- 模型加载 ---------------------
model = RestormerModel().load_from_checkpoint(ckpt_path, strict=True).to(device)
model.eval()

# --------------------- 特征提取 ---------------------
features, labels, file_paths = [], [], []

with torch.no_grad():
    for images, lbls in dataloader:
        images = images.to(device)
        feats = model.extract_encoder_features(images)
        features.append(feats.cpu().numpy())
        labels.extend(lbls.numpy())
        batch_paths = [dataset.samples[i][0] for i in range(len(file_paths), len(file_paths) + len(lbls))]
        file_paths.extend(batch_paths)

features = np.concatenate(features, axis=0)
labels = np.array(labels)

# --------------------- PCA + t-SNE ---------------------
if features.shape[1] > 50:
    features = PCA(n_components=50).fit_transform(features)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(features)

# --------------------- 离群点检测 ---------------------
outlier_indices, filtered_indices = [], []
num_classes = len(dataset.classes)

for i in range(num_classes):
    idxs = np.where(labels == i)[0]
    class_points = tsne_results[idxs]
    mean = np.mean(class_points, axis=0)
    std = np.std(class_points, axis=0)
    dists = np.linalg.norm(class_points - mean, axis=1)
    threshold = 3 * np.mean(std)
    class_outliers = idxs[dists > threshold]
    class_inliers = idxs[dists <= threshold]
    outlier_indices.extend(class_outliers)
    filtered_indices.extend(class_inliers)

# --------------------- 保存离群样本路径 ---------------------
outlier_paths = [file_paths[i] for i in outlier_indices]
os.makedirs(os.path.dirname(outlier_save_txt), exist_ok=True)
with open(outlier_save_txt, 'w') as f:
    for p in outlier_paths:
        f.write(p + '\n')

print(f"🔍 共检测到 {len(outlier_indices)} 个离群点，已保存至 {outlier_save_txt}")

# --------------------- 绘制可视化 ---------------------
plt.figure(figsize=(10, 8))
colors = cm.rainbow(np.linspace(0, 1, num_classes))

# 绘制正常点
for i, c in enumerate(colors):
    idxs = labels[filtered_indices] == i
    plt.scatter(tsne_results[filtered_indices][idxs, 0],
                tsne_results[filtered_indices][idxs, 1],
                color=c, label=dataset.classes[i], alpha=0.65, s=45, edgecolor='white', linewidth=0.5)

# 绘制离群点（红叉，可选）
# plt.scatter(tsne_results[outlier_indices, 0],
#             tsne_results[outlier_indices, 1],
#             color='red', marker='x', s=40, label='Outliers', linewidths=1.2)

# 去掉坐标轴刻度与边界
plt.axis('off')
plt.xticks([])
plt.yticks([])

# --------------------- 设置图形边框（加阴影感） ---------------------
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(3)
    spine.set_color('#222222')  # 更深的黑色边框

# 添加白色背景与轻微阴影
plt.gcf().patch.set_facecolor('white')
plt.gca().set_facecolor('#f8f8f8')
plt.gca().patch.set_alpha(1.0)

# --------------------- 图例设置（放大 + 外框） ---------------------
legend = plt.legend(
    loc='lower center',           # 底部居中
    bbox_to_anchor=(0.5, -0.12),  # 图例位置下移
    ncol=min(5, num_classes),     # 横向排列
    fontsize=14,                  # 字号再放大
    frameon=True,                 # 添加外框
    fancybox=True,                # 圆角边框
    shadow=True,                  # 添加阴影
    edgecolor='black',            # 外框颜色
    title="Image Classes",        # 添加标题
    title_fontsize=14             # 标题字体
)

# 调整图例背景色
legend.get_frame().set_facecolor('#f0f0f0')
legend.get_frame().set_linewidth(1.5)

# --------------------- 保存结果 ---------------------
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.3, facecolor='white')
plt.show()