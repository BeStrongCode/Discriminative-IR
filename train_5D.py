import argparse
import subprocess

from torch.xpu import device
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from test import test_Denoise
from utils.dataset_utils import PromptTrainDataset, DenoiseTestDataset, DerainDehazeDataset, PromptTrainDataset5D
from net.model import Restormer_orgin
from net.PromptIR import PromptIR
from net.model_zd_zc import Restormer
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890/"
os.environ["https_proxy"] = "http://127.0.0.1:7890/"
# wandb.login(key='403a08bb112279b0df3eeaf16e08e02f8e5fc634')
# os.environ["WANDB_MODE"] = "offline"
from options_5D import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from utils.val_utils import AverageMeter, compute_psnr_ssim
import sys
from pytorch_lightning.utilities.rank_zero import rank_zero_only
# torch.set_float32_matmul_precision('medium')

def test_Denoise(net, dataset, sigma=15):


    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored,_ = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
        return psnr.avg,ssim.avg


def test_Derain_Dehaze(net, dataset, task="derain"):

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored,_ = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
        print(f"{task}:PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg,ssim.avg

def test_Deblur(net, dataset, task="deblur"):

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored,_ = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
        print(f"{task}:PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg,ssim.avg
        

def test_Lowlight(net, dataset, task="lowlight"):

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored,_ = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
        print(f"{task}:PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg,ssim.avg
# def test_Denoise(model, dataset, sigma=15):
#     # output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
#     # subprocess.check_output(['mkdir', '-p', output_path])
#     #
#
#     dataset.set_sigma(sigma)
#     testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
#
#     psnr = AverageMeter()
#     ssim = AverageMeter()
#
#     with torch.no_grad():
#         for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
#             input_, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
#             height, width = input_.shape[2], input_.shape[3]
#             img_multiple_of = 8
#             H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
#                         (width + img_multiple_of) // img_multiple_of) * img_multiple_of
#             padh = H - height if height % img_multiple_of != 0 else 0
#             padw = W - width if width % img_multiple_of != 0 else 0
#             input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
#
#             # restored, _ = net(degrad_patch)
#
#             # test the image tile by tile
#             b, c, h, w = input_.shape
#             tile = min(128, h, w)
#             assert tile % 8 == 0, "tile size should be multiple of 8"
#             tile_overlap = 32
#
#             stride = tile - tile_overlap
#             h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
#             w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
#             E = torch.zeros(b, c, h, w).type_as(input_)
#             W = torch.zeros_like(E)
#
#             for h_idx in h_idx_list:
#                 for w_idx in w_idx_list:
#                     in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
#                     out_patch, _ = model(in_patch)
#                     out_patch_mask = torch.ones_like(out_patch)
#
#                     E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
#                     W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
#             restored = E.div_(W)
#
#             restored = torch.clamp(restored, 0, 1)
#
#             # Unpad the output
#             restored = restored[:, :, :height, :width]
#
#
#             temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
#
#             psnr.update(temp_psnr, N)
#             ssim.update(temp_ssim, N)
#             # save_image_tensor(restored, output_path + clean_name[0] + '.png')
#
#         print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
#         return psnr.avg, ssim.avg
#
#
# def test_Derain_Dehaze(model, dataset, task="derain"):
#     # output_path = testopt.output_path + task + '/'
#     # subprocess.check_output(['mkdir', '-p', output_path])
#     #
#     dataset.set_dataset(task)
#     testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
#
#     psnr = AverageMeter()
#     ssim = AverageMeter()
#
#     with torch.no_grad():
#         for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
#             input_, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
#
#
#             height, width = input_.shape[2], input_.shape[3]
#             img_multiple_of = 8
#             H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
#                         (width + img_multiple_of) // img_multiple_of) * img_multiple_of
#             padh = H - height if height % img_multiple_of != 0 else 0
#             padw = W - width if width % img_multiple_of != 0 else 0
#             input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
#
#
#             # test the image tile by tile
#             b, c, h, w = input_.shape
#             tile = min(128, h, w)
#             assert tile % 8 == 0, "tile size should be multiple of 8"
#             tile_overlap = 32
#
#             stride = tile - tile_overlap
#             h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
#             w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
#             E = torch.zeros(b, c, h, w).type_as(input_)
#             W = torch.zeros_like(E)
#
#             for h_idx in h_idx_list:
#                 for w_idx in w_idx_list:
#                     in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
#                     out_patch, _ = model(in_patch)
#                     out_patch_mask = torch.ones_like(out_patch)
#
#                     E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
#                     W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
#             restored = E.div_(W)
#
#             restored = torch.clamp(restored, 0, 1)
#
#             # Unpad the output
#             restored = restored[:, :, :height, :width]
#
#
#             temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
#             psnr.update(temp_psnr, N)
#             ssim.update(temp_ssim, N)
#             #
#             # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
#         print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
#         return psnr.avg, ssim.avg




class ManualEvalCallback(Callback):
    def __init__(self, testopt, result_txt):
        super().__init__()
        self.testopt = testopt
        self.denoise_testset = DenoiseTestDataset(testopt)
        self.derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        self.result_txt = result_txt

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < 100:
            if trainer.current_epoch % 5 != 0:
                return  # 只在每5个epoch执行一次测试和权重保存

        if trainer.global_rank != 0:
            return  # 仅在主进程执行一次
        print(f"[Epoch {trainer.current_epoch}] Running manual test...")
        pl_module.eval()
        avg_psnr,avg_ssim = 0,0
        with torch.no_grad():


            psnr,ssim = test_Denoise(pl_module, self.denoise_testset, sigma=25)
            # wandb.log({f"Denoise_PSNR_sigma{25}": psnr,
            #            f"Denoise_SSIM_sigma{25}": ssim,
            #            })
            avg_psnr += psnr
            avg_ssim += ssim




            psnr,ssim = test_Derain_Dehaze(pl_module, self.derain_set, task="derain")
            # wandb.log({f"Derain_PSNR": psnr,
            #            f"Derain_SSIM": ssim,
            #            })
            avg_psnr += psnr
            avg_ssim += ssim

            psnr,ssim = test_Derain_Dehaze(pl_module, self.derain_set, task="dehaze")
            # wandb.log({f"Dehaze_PSNR": psnr,
            #            f"Dehaze_SSIM": ssim,
            #            })
            avg_psnr += psnr
            avg_ssim += ssim

            # psnr,ssim = test_Deblur(pl_module, self.derain_set, task="deblur")
            # avg_psnr += psnr
            # avg_ssim += ssim

            psnr,ssim = test_Lowlight(pl_module, self.derain_set, task="lowlight")
            avg_psnr += psnr
            avg_ssim += ssim

        print(f"{trainer.current_epoch}:avg_psnr:{avg_psnr/4},avg_ssim:{avg_ssim/4}")
        # wandb.log({f"avg_psnr":avg_psnr/5,
        #           f"avg_ssim":avg_ssim/5,
        #           })
        with open(self.result_txt,'a') as f:
            # 按格式写入 epoch、avg_psnr、avg_ssim，用制表符或逗号分隔
            f.write(f"{trainer.current_epoch}\t{avg_psnr / 4:.4f}\t{avg_ssim / 4:.4f}\n")
        pl_module.train()


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()

    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, _  = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)


        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.l1_loss(y_hat, y)
    #     self.log("val_loss", loss, prog_bar=True, sync_dist=True)  # 必须 log 出来
    #     return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_last_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]
    def extract_encoder_features(self, x):
        # 这是你要新增的方法
        _, latent = self.net(x)
        # feat 的 shape 应该是 [B, C, H, W]，你可以选择平均池化或 flatten 处理成 [B, C] 向量
        feat = torch.mean(latent, dim=[2, 3])  # global average pooling
        return feat




class RestormerModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Restormer()
        self.loss_fn = nn.L1Loss()
        # self.criterion = nn.CrossEntropyLoss()
        # 冻结编码层
        # self.freeze_encoder_layer()
    def forward(self, x):
        restored = self.net(x)
        return restored

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored,_ = self.net(degrad_patch)
        # label = de_id
        restor_loss = self.loss_fn(restored, clean_patch)
        # cls_loss = self.criterion(cls,label)
        loss = restor_loss



        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # self.log("restor_loss", restor_loss)
        # self.log("cls_loss", cls_loss)

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_last_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]

    def extract_encoder_features(self, x):
        # 这是你要新增的方法
        _, latent = self.net(x)
        # feat 的 shape 应该是 [B, C, H, W]，你可以选择平均池化或 flatten 处理成 [B, C] 向量
        feat = torch.mean(latent, dim=[2, 3])  # global average pooling
        return feat




def main():
    print("Options")
    print(opt)
    # if opt.wblogger is not None:
    #     logger  = WandbLogger(project=opt.wblogger,name="Restormer-gumbel1+maxpool+3pgm")
    # else:
    #     logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset5D(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    result_txt = os.path.join(opt.ckpt_dir,'results.txt')
    with open(result_txt,'w') as file:
        file.write(f'epoch\tavg_psnr\tavg_ssim\n')

    # fintune
    # ckpt_path = '/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/train_ckpt/5D/restormer_zd_zc/epoch=119-step=277080.ckpt'
    # 加载 Restormer 权重
    ckpt_path = "/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/Pretrain/5D/restormer_zd_zc_AdamW/checkpoint_epoch_V1_0010_clsloss1.6110_l1loss0.0000.pth.tar"  # 你的 Restormer 预训练权重路径
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = state_dict['state_dict']

    # fintune
    # model = RestormerModel().load_from_checkpoint(ckpt_path).cuda()
    # model = PromptIRModel().cuda()
    model = RestormerModel().cuda()
    # 只加载 net 部分
    model.net.load_state_dict(state_dict, strict=False)


    parser = argparse.ArgumentParser()
    parser.add_argument('--denoise_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/CBSD68/')
    parser.add_argument('--derain_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/Rain100L/')
    parser.add_argument('--dehaze_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/')
    parser.add_argument('--deblur_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/deblurring/GOPRO_Large/test1/')
    parser.add_argument('--lowlight_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/low-light/LOLdataset/test/')
    testopt = parser.parse_args()



    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",callbacks=[checkpoint_callback,ManualEvalCallback(testopt,result_txt)])
    # trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus, strategy="ddp_find_unused_parameters_true",callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



