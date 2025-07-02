import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from test import test_Denoise
from utils.dataset_utils import PromptTrainDataset, DenoiseTestDataset, DerainDehazeDataset
from net.model import PromptIR
from net.model import Restormer
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890/"
os.environ["https_proxy"] = "http://127.0.0.1:7890/"
wandb.login(key='403a08bb112279b0df3eeaf16e08e02f8e5fc634')
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from utils.val_utils import AverageMeter, compute_psnr_ssim
import sys


def test_Denoise(net, dataset, sigma=15):


    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
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

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
        print(f"{task}:PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg,ssim.avg


class ManualEvalCallback(Callback):
    def __init__(self, testopt):
        super().__init__()
        self.testopt = testopt
        self.denoise_testset = DenoiseTestDataset(testopt)
        self.derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 != 0:
            return  # 只在每5个epoch执行一次测试和权重保存


        print(f"[Epoch {trainer.current_epoch}] Running manual test...")
        pl_module.eval()
        avg_psnr,avg_ssim = 0,0
        with torch.no_grad():
            psnr,ssim = test_Denoise(pl_module, self.denoise_testset, sigma=15)
            wandb.log({f"Denoise_PSNR_sigma{15}": psnr,
                       f"Denoise_SSIM_sigma{15}": ssim,
                         })
            avg_psnr += psnr
            avg_ssim += ssim

            psnr,ssim = test_Denoise(pl_module, self.denoise_testset, sigma=25)
            wandb.log({f"Denoise_PSNR_sigma{25}": psnr,
                       f"Denoise_SSIM_sigma{25}": ssim,
                       })
            avg_psnr += psnr
            avg_ssim += ssim

            psnr,ssim = test_Denoise(pl_module, self.denoise_testset, sigma=50)
            wandb.log({f"Denoise_PSNR_sigma{50}": psnr,
                       f"Denoise_SSIM_sigma{50}": ssim,
                       })
            avg_psnr += psnr
            avg_ssim += ssim


            psnr,ssim = test_Derain_Dehaze(pl_module, self.derain_set, task="derain")
            wandb.log({f"Derain_PSNR": psnr,
                       f"Derain_SSIM": ssim,
                       })
            avg_psnr += psnr
            avg_ssim += ssim

            psnr,ssim = test_Derain_Dehaze(pl_module, self.derain_set, task="dehaze")
            wandb.log({f"Dehaze_PSNR": psnr,
                       f"Dehaze_SSIM": ssim,
                       })
            avg_psnr += psnr
            avg_ssim += ssim
        print(f"{trainer.current_epoch}:avg_psnr:{avg_psnr/5},avg_ssim:{avg_ssim/5}")
        wandb.log({f"avg_psnr":avg_psnr/5,
                  f"avg_ssim":avg_ssim/5,
                  })

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
        restored = self.net(degrad_patch)

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
        # 冻结编码层
        self.freeze_encoder_layer()
    def forward(self, x):
        restored = self.net(x)
        return restored

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        restor_loss = self.loss_fn(restored, clean_patch)
        loss = restor_loss


        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)

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

    def freeze_encoder_layer(self):
        # 冻结所有相关层的参数
        for param in self.net.patch_embed.parameters():
            param.requires_grad = False
        for param in self.net.encoder_level1.parameters():
            param.requires_grad = False
        for param in self.net.encoder_level2.parameters():
            param.requires_grad = False
        for param in self.net.encoder_level3.parameters():
            param.requires_grad = False
        for param in self.net.latent.parameters():
            param.requires_grad = False


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="Restormer-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    pre_train_path = 'train_ckpt/restormer***/epoch=119-step=515760.ckpt'
    model = RestormerModel().load_from_checkpoint(pre_train_path,strict=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--denoise_path', type=str, default='/media/rqy/hhh/all-in-one-image-restoration/dataset/denoising/CBSD68/')
    parser.add_argument('--derain_path', type=str, default='/media/rqy/hhh/all-in-one-image-restoration/dataset/deraining/Rain100L/')
    parser.add_argument('--dehaze_path', type=str, default='/media/rqy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/')
    testopt = parser.parse_args()

    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback,ManualEvalCallback(testopt)])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



