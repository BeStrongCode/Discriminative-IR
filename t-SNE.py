import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from train import RestormerModel, PromptIRModel

# 图像预处理（与预训练模型匹配）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
    #                      std=[0.229, 0.224, 0.225])  # ImageNet 方差
])
ckpt_path = 'train_ckpt/restormer**/epoch=119-step=389760.ckpt'
# 加载图像文件夹
dataset_path = "images"  # 修改为你图片根目录路径
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RestormerModel().load_from_checkpoint(ckpt_path).cuda()

model.eval()
features = []
labels = []

with torch.no_grad():
    for images, lbls in dataloader:
        images = images.to(device)
        feats = model.extract_encoder_features(images)  # shape: [B, 512]
        features.append(feats.cpu().numpy())
        labels.extend(lbls.numpy())
save_path = 't-SNE/de_t-SNE_restormer**.png'
features = np.concatenate(features, axis=0)
labels = np.array(labels)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(features)
import matplotlib.cm as cm

plt.figure(figsize=(10, 8))
num_classes = len(dataset.classes)
colors = cm.rainbow(np.linspace(0, 1, num_classes))

for i, c in enumerate(colors):
    idxs = labels == i
    plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], color=c, label=dataset.classes[i], alpha=0.6)

plt.legend()
plt.title("t-SNE Visualization of Degradation images")

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()