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
# wandb.login(key='403a08bb112279b0df3eeaf16e08e02f8e5fc634')
os.environ["WANDB_MODE"] = "offline"
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from utils.val_utils import AverageMeter, compute_psnr_ssim
import sys
from pytorch_lightning.utilities.rank_zero import rank_zero_only


# ------------------------------ 工具函数：识别/保存/加载 encoder 参数 ------------------------------
def is_encoder_param(name: str) -> bool:
    """根据名字粗略判断是否属于encoder。可按你工程实际进一步收紧。"""
    lname = name.lower()
    return ("encoder" in lname) or (".enc" in lname) or ("down" in lname and "block" in lname)


def extract_encoder_state_dict(full_state: dict) -> dict:
    """尽量只提取属于encoder的键；若没匹配到，则回退为所有键（用于宽松加载+strict=False）。"""
    enc = {k: v for k, v in full_state.items() if is_encoder_param(k)}
    if len(enc) == 0:
        # 回退：给出提示，并返回全量；加载时 strict=False 只会匹配上相同名字的层
        print("[WARN] No encoder-like keys matched; will fallback to full state_dict for flexible loading.")
        return full_state
    return enc


def load_encoder_weights_into_model(model: nn.Module, encoder_ckpt_path: str):
    ckpt = torch.load(encoder_ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    enc_state = extract_encoder_state_dict(state)
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    print(f"[Load Encoder] Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")


# ------------------------------ 阶段A：Encoder多分类训练 ------------------------------
class RestormerEncoderClassifier(pl.LightningModule):
    """
    使用 Restormer 作为特征提取器，拿 latent 做 GAP + 线性分类。
    只训练 encoder(+分类头)，decoder 尽量冻结（若能匹配到）。
    """
    def __init__(self, num_classes: int = 3, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = Restormer()
        # 分类头：latent -> GAP -> fc
        self.cls_head = nn.Linear(256, num_classes, bias=True)  # 256是示例，改成你latent的通道数

        # 自动冻结 decoder（若能匹配到），放行 encoder
        for n, p in self.net.named_parameters():
            if "decoder" in n.lower() or ("up" in n.lower() and "block" in n.lower()):
                p.requires_grad = False  # 冻结可能的解码器部分

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # 返回分类 logits
        restored, latent = self.net(x)  # latent: [B,C,H,W]
        feat = torch.mean(latent, dim=[2, 3])  # GAP -> [B,C]
        logits = self.cls_head(feat)
        return logits

    def training_step(self, batch, batch_idx):
        ([_, de_id], degrad_patch, _) = batch
        logits = self.forward(degrad_patch)
        loss = self.criterion(logits, de_id.long())
        acc = (logits.argmax(dim=1) == de_id).float().mean()
        self.log_dict({"cls_loss": loss, "cls_acc": acc}, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 仅优化需要梯度的参数（即 encoder + cls_head）
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.hparams.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5, max_epochs=opt.epochs)
        return [optimizer], [scheduler]

    def on_train_end(self):
        # 仅导出 Encoder（若能匹配到），否则导出全量（供 strict=False 兼容加载）
        state = self.net.state_dict()
        enc_state = extract_encoder_state_dict(state)
        save_path = os.path.join(opt.ckpt_dir, "encoder_cls.pth")
        torch.save(enc_state, save_path)
        print(f"[Stage-A Saved] Encoder weights saved to: {save_path}")




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
        restored, _ = self.net(degrad_patch)
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
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="Restormer-gumbel1+maxpool+3pgm")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    result_txt = os.path.join(opt.ckpt_dir, 'results.txt')
    with open(result_txt, 'w') as file:
        file.write(f'epoch\tavg_psnr\tavg_ssim\n')

    # ckpt_path = 'train_ckpt/restormer_shufprompt/epoch=88-step=382522.ckpt'
    model = RestormerEncoderClassifier(num_classes=5,lr=2e-4)


    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, strategy="ddp", logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



