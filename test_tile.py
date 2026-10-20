import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

from train import RestormerModel
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F


def test_Denoise(model, dataset, sigma=15):
    # output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    # subprocess.check_output(['mkdir', '-p', output_path])
    #

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            input_, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            height, width = input_.shape[2], input_.shape[3]
            img_multiple_of = 8
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            # restored, _ = net(degrad_patch)

            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(128, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = 32

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch, _ = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :height, :width]


            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            # save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg


def test_Derain_Dehaze(model, dataset, task="derain"):
    # output_path = testopt.output_path + task + '/'
    # subprocess.check_output(['mkdir', '-p', output_path])
    #
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            input_, clean_patch = degrad_patch.cuda(), clean_patch.cuda()


            height, width = input_.shape[2], input_.shape[3]
            img_multiple_of = 8
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                        (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')


            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(128, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = 32

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch, _ = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :height, :width]


            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            #
            # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
        return psnr.avg, ssim.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str,
                        default="/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/",
                        help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str,
                        default="/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/",
                        help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str,
                        default="/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/",
                        help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="train_ckpt/Restormer-fftprompt/epoch=119-step=193440.ckpt",
                        help='checkpoint save path')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_name

    denoise_splits = ["CBSD68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path, i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))

    # net  = PromptIRModel().load_from_checkpoint(ckpt_path).cuda()
    net = RestormerModel().load_from_checkpoint(ckpt_path).cuda()
    net.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        if testopt.mode == 0:
            for testset, name in zip(denoise_tests, denoise_splits):
                print('Start {} testing Sigma=15...'.format(name))
                test_Denoise(net, testset, sigma=15)

                print('Start {} testing Sigma=25...'.format(name))
                test_Denoise(net, testset, sigma=25)

                print('Start {} testing Sigma=50...'.format(name))
                test_Denoise(net, testset, sigma=50)
        elif testopt.mode == 1:
            print('Start testing rain streak removal...')
            derain_base_path = testopt.derain_path
            for name in derain_splits:
                print('Start testing {} rain streak removal...'.format(name))
                testopt.derain_path = os.path.join(derain_base_path, name)
                derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
                test_Derain_Dehaze(net, derain_set, task="derain")
        elif testopt.mode == 2:
            print('Start testing SOTS...')
            derain_base_path = testopt.derain_path
            name = derain_splits[0]
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="dehaze")
        elif testopt.mode == 3:
            for testset, name in zip(denoise_tests, denoise_splits):
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
                testopt.derain_path = os.path.join(derain_base_path, name)
                derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
                psnr, ssim = test_Derain_Dehaze(net, derain_set, task="derain")
                total_psnr += psnr
                total_ssim += ssim

            print('Start testing SOTS...')
            psnr, ssim = test_Derain_Dehaze(net, derain_set, task="dehaze")
            total_psnr += psnr
            total_ssim += ssim

            print(f'avg_psnr:{total_psnr / 5}, avg_ssim:{total_ssim / 5}')
