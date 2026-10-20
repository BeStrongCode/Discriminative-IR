import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

from train import RestormerModel
from train_5D import test_Deblur, test_Lowlight
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.PromptIR import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F

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
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def test_Denoise(net, dataset, sigma=15):
    output_path = os.path.join(testopt.output_path, 'denoise' + str(sigma))
    os.makedirs(output_path,exist_ok=True)
    # subprocess.check_output(['mkdir', '-p', output_path])
    # #

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
            save_image_tensor(restored, os.path.join(output_path, clean_name[0] + f'_{temp_psnr:.2f}_{temp_ssim:.2f}.png'))

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg



def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = os.path.join(testopt.output_path, task)
    os.makedirs(output_path,exist_ok=True)
    #
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
            newname = os.path.join(output_path, degraded_name[0] + f'_{temp_psnr:.2f}_{temp_ssim:.2f}.png')
            save_image_tensor(restored, newname)
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=4,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one, 4 for 5D')

    parser.add_argument('--denoise_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/", help='save path of test hazy images')
    parser.add_argument('--deblur_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/deblurring/GOPRO_Large/test1/')
    parser.add_argument('--lowlight_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/low-light/LOLdataset/test/')
    parser.add_argument('--output_path', type=str, default="output/test/PromptIR_balence_data", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="train_ckpt/5D/PromptIR_balence_data/epoch=119-step=277080.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path =  testopt.ckpt_name

    
    
    denoise_splits = ["CBSD68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)


    print("CKPT name : {}".format(ckpt_path))

    net  = PromptIRModel().load_from_checkpoint(ckpt_path).cuda()
    # net  = RestormerModel().load_from_checkpoint(ckpt_path).cuda()
    net.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        if testopt.mode == 0:
            for testset,name in zip(denoise_tests,denoise_splits) :
                print('Start {} testing Sigma=15...'.format(name))
                test_Denoise(net, testset, sigma=15)

                print('Start {} testing Sigma=25...'.format(name))
                test_Denoise(net, testset, sigma=25)

                print('Start {} testing Sigma=50...'.format(name))
                test_Denoise(net, testset, sigma=50)

                print('Start {} testing Sigma=100...'.format(name))
                test_Denoise(net, testset, sigma=100)
        elif testopt.mode == 1:
            print('Start testing rain streak removal...')
            derain_base_path = testopt.derain_path
            for name in derain_splits:
                print('Start testing {} rain streak removal...'.format(name))
                testopt.derain_path = os.path.join(derain_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
                test_Derain_Dehaze(net, derain_set, task="derain")
        elif testopt.mode == 2:
            print('Start testing SOTS...')
            derain_base_path = testopt.derain_path
            name = derain_splits[0]
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="dehaze")
        elif testopt.mode == 3:
            for testset,name in zip(denoise_tests,denoise_splits) :
                print('Start {} testing Sigma=15...'.format(name))
                psnr, ssim = test_Denoise(net, testset, sigma=15)
                total_psnr += psnr
                total_ssim += ssim

                print('Start {} testing Sigma=25...'.format(name))
                psnr, ssim = test_Denoise(net, testset, sigma=25)
                total_psnr += psnr
                total_ssim += ssim

                print('Start {} testing Sigma=50...'.format(name))
                psnr, ssim = test_Denoise(net, testset, sigma=50)
                total_psnr += psnr
                total_ssim += ssim


            derain_base_path = testopt.derain_path
            print(derain_splits)
            for name in derain_splits:

                print('Start testing {} rain streak removal...'.format(name))
                testopt.derain_path = os.path.join(derain_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
                psnr, ssim = test_Derain_Dehaze(net, derain_set, task="derain")
                total_psnr += psnr
                total_ssim += ssim


            print('Start testing SOTS...')
            psnr, ssim = test_Derain_Dehaze(net, derain_set, task="dehaze")
            total_psnr += psnr
            total_ssim += ssim

            print(f'avg_psnr:{total_psnr/5}, avg_ssim:{total_ssim/5}')
        elif testopt.mode == 4:

            for testset,name in zip(denoise_tests,denoise_splits) :
                # print('Start {} testing Sigma=15...'.format(name))
                # psnr, ssim = test_Denoise(net, testset, sigma=15)
                # total_psnr += psnr
                # total_ssim += ssim

                print('Start {} testing Sigma=25...'.format(name))
                psnr, ssim = test_Denoise(net, testset, sigma=25)
                total_psnr += psnr
                total_ssim += ssim

                # print('Start {} testing Sigma=50...'.format(name))
                # psnr, ssim = test_Denoise(net, testset, sigma=50)
                # total_psnr += psnr
                # total_ssim += ssim


            derain_base_path = testopt.derain_path
            print(derain_splits)

            for name in derain_splits:

                print('Start testing {} rain streak removal...'.format(name))
                testopt.derain_path = os.path.join(derain_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
                psnr, ssim = test_Derain_Dehaze(net, derain_set, task="derain")
                total_psnr += psnr
                total_ssim += ssim



            print('Start testing SOTS...')
            psnr, ssim = test_Derain_Dehaze(net, derain_set, task="dehaze")
            total_psnr += psnr
            total_ssim += ssim


            print('Start testing Gopro...')
            psnr,ssim = test_Derain_Dehaze(net, derain_set, task="deblur")
            total_psnr += psnr
            total_ssim += ssim

            print('Start testing LOL...')
            psnr,ssim = test_Derain_Dehaze(net, derain_set, task="lowlight")
            total_psnr += psnr
            total_ssim += ssim

        print(f'avg_psnr:{total_psnr/5}, avg_ssim:{total_ssim/5}')
