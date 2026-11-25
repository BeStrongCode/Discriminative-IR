

import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.dataset_utils import PromptTrainDataset, DenoiseTestDataset, DerainDehazeDataset

from net.PromptIR import PromptIR
from net.model import DisIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890/"
os.environ["https_proxy"] = "http://127.0.0.1:7890/"

from options import options as opt
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

class DegradationClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=5, hidden_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, latent):
        return self.head(latent)


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
            psnr,ssim = test_Denoise(pl_module, self.denoise_testset, sigma=15)
            # wandb.log({f"Denoise_PSNR_sigma{15}": psnr,
            #            f"Denoise_SSIM_sigma{15}": ssim,
            #              })
            avg_psnr += psnr
            avg_ssim += ssim

            psnr,ssim = test_Denoise(pl_module, self.denoise_testset, sigma=25)
            # wandb.log({f"Denoise_PSNR_sigma{25}": psnr,
            #            f"Denoise_SSIM_sigma{25}": ssim,
            #            })
            avg_psnr += psnr
            avg_ssim += ssim

            psnr,ssim = test_Denoise(pl_module, self.denoise_testset, sigma=50)
            # wandb.log({f"Denoise_PSNR_sigma{50}": psnr,
            #            f"Denoise_SSIM_sigma{50}": ssim,
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
        print(f"{trainer.current_epoch}:avg_psnr:{avg_psnr/5},avg_ssim:{avg_ssim/5}")
        # wandb.log({f"avg_psnr":avg_psnr/5,
        #           f"avg_ssim":avg_ssim/5,
        #           })
        with open(self.result_txt,'a') as f:
            # 按格式写入 epoch、avg_psnr、avg_ssim，用制表符或逗号分隔
            f.write(f"{trainer.current_epoch}\t{avg_psnr / 5:.4f}\t{avg_ssim / 5:.4f}\n")
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
    def __init__(self, use_degradation_cls=False, cls_loss_weight=0.1, num_degradation_classes=5):
        super().__init__()
 
        self.save_hyperparameters()
        self.use_degradation_cls = use_degradation_cls
        self.cls_loss_weight = cls_loss_weight

        self.net = DisIR()
        self.loss_fn = nn.L1Loss()

        self.cls_head = None
        if self.use_degradation_cls:
            # Both backbones feed the returned latent into the first decoder upsample.
            latent_channels = self.net.up4_3.body[0].in_channels
            self.cls_head = DegradationClassifier(latent_channels, num_degradation_classes)
            self.criterion = nn.CrossEntropyLoss()

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
        restored, latent = self.net(degrad_patch)
        restor_loss = self.loss_fn(restored, clean_patch)
        loss = restor_loss

        if self.use_degradation_cls:
            logits = self.cls_head(latent)
            labels = torch.as_tensor(de_id, device=logits.device, dtype=torch.long).view(-1)
            cls_loss = self.criterion(logits, labels)
            loss = restor_loss + self.cls_loss_weight * cls_loss
            self.log("cls_loss", cls_loss)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("restor_loss", restor_loss)

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

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    result_txt = os.path.join(opt.ckpt_dir,'results.txt')
    with open(result_txt,'w') as file:
        file.write(f'epoch\tavg_psnr\tavg_ssim\n')

    # ckpt_path = '/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/train_ckpt/3D/Restormer-max_zdzc_AdamW10/epoch=115-32.6635_0.9176.ckpt'
    # 加载 Restormer 权重
    # ckpt_path = "/media/hy/hhh/all-in-one-image-restoration/PromptIR-main/Pretrain/3D/PromptIR/checkpoint_epoch_V1_0010_loss0.0002.pth.tar"  # 你的 Restormer 预训练权重路径
    # state_dict = torch.load(ckpt_path, map_location="cpu")
    # state_dict = state_dict['state_dict']

    model = RestormerModel(
        use_degradation_cls=opt.use_degradation_cls,
        cls_loss_weight=opt.cls_loss_weight,
        num_degradation_classes=opt.num_degradation_classes,
    ).cuda()
    # 只加载 net 部分
    # model.net.load_state_dict(state_dict, strict=True)
    # model.net.prompt3.prompt_param.requires_grad = False

    # 微调
    # model = RestormerModel().load_from_checkpoint(ckpt_path).cuda()

    testopt = opt



    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",callbacks=[checkpoint_callback,ManualEvalCallback(testopt,result_txt)]) # strategy="ddp"
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



