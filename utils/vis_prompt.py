import random

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
import os

from tqdm import tqdm

from train import PromptIRModel
from utils.image_utils import crop_img


def byone():
    image_path = '../images/hazy/0034_0.9_0.16.jpg'

    degrad = crop_img(np.array(Image.open(image_path).convert('RGB')), base=16)
    degrad = degrad.astype(np.float32) / 255.0
    degrad = torch.from_numpy(degrad).cuda()
    degrad = degrad.permute(2,0,1).unsqueeze(0)

    save_path = os.path.join('../vis_results',image_path.split('/')[2])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    newimg_path = os.path.join(save_path,image_path.split('/')[-1])


    ckpt_path = '../ckpt/model.ckpt'
    model = PromptIRModel().load_from_checkpoint(ckpt_path).cuda()
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

def plotweights(weights,save_path):
    # 创建图形和3D坐标轴
    weights = np.array(weights)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建x, y的坐标，z为0，柱子的宽度和深度都设置为1
    x = np.arange(5)
    y = np.arange(weights.shape[0])
    x, y = np.meshgrid(x, y)

    # 设定柱子的宽度和深度
    dx = np.ones_like(x)
    dy = np.ones_like(y)
    dz = weights.flatten()  # 展平weight1，作为每个柱子的高度

    # 为每个柱子指定不同的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, 5))  # 使用viridis色图为每个柱子分配不同的颜色

    # 绘制三维柱状图，并为每个prompt type指定不同颜色
    for i in range(5):
        ax.bar3d(x.flatten()[i::5], y.flatten()[i::5], np.zeros_like(dz)[i::5],
                 dx.flatten()[i::5], dy.flatten()[i::5], dz[i::5], color=colors[i])

    # 设置标签
    ax.set_xlabel('prompt type')
    ax.set_ylabel('test samples')
    ax.set_zlabel('value')


    # 设置x轴和y轴间隔
    ax.set_xticks(np.arange(5) + 0.5)  # 在x轴上设置间隔
    ax.set_yticks(np.arange(weights.shape[0]) + 0.5)  # 在y轴上设置间隔
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # 显示图形
    # plt.show()


def byfolder():
    type = 'rainy'
    img_folder = os.path.join('../images',type)
    ckpt_path = '../ckpt/model.ckpt'
    model = PromptIRModel().load_from_checkpoint(ckpt_path).cuda()
    model.eval()
    weight1,weight2,weight3 = [],[],[]
    with torch.no_grad():
        for folder,_,img_list in os.walk(img_folder):
            for name in tqdm(random.sample(img_list,50)):
                img_file = os.path.join(folder,name)
                img = crop_img(np.array(Image.open(img_file).convert('RGB')), base=16)
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).cuda()
                img = img.permute(2,0,1).unsqueeze(0)
                _, weights = model(img)
                weights = np.array(weights.cpu())
                weight1.append(weights[0])
                weight2.append(weights[1])
                weight3.append(weights[2])

    plotweights(weight1,'weight1.png')
    plotweights(weight2,'weight2.png')
    plotweights(weight3,'weight3.png')

byfolder()