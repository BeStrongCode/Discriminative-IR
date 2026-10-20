import random

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
import os

from tqdm import tqdm

from train import PromptIRModel, RestormerModel
from utils.image_utils import crop_img




def byone():
    image_path = '../images/rainy/rain-005.png'

    degrad = crop_img(np.array(Image.open(image_path).convert('RGB')), base=16)
    degrad = degrad.astype(np.float32) / 255.0
    degrad = torch.from_numpy(degrad).cuda()
    degrad = degrad.permute(2,0,1).unsqueeze(0)

    save_path = os.path.join('../vis_results',image_path.split('/')[2])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    newimg_path = os.path.join(save_path,image_path.split('/')[-1])


    ckpt_path = '../train_ckpt/restormer_shufprompt1/epoch=110-step=178932.ckpt'
    model = RestormerModel().load_from_checkpoint(ckpt_path).cuda()
    model.eval()
    with torch.no_grad():

        res,weights = model(degrad)
        weights = weights.cpu()
        data = np.array(weights)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 获取 x, y 的坐标索引
        xpos, ypos = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        # 柱子的尺寸
        dx = dy = 0.5
        dz = data.flatten()

        # 绘制柱状图
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

        # 设置坐标轴标签
        ax.set_xlabel('Vector Index')
        ax.set_ylabel('Group Index')
        ax.set_zlabel('Value')

        plt.title('3D Bar Plot of (3,5) Vectors')
        plt.savefig(newimg_path, dpi=300, bbox_inches='tight')

        plt.show()


byone()