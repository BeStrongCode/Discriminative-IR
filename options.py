import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=300, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.') # 2e-4

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')


## add degradation class loss
parser.add_argument('--use_degradation_cls', action='store_true',
                    help='enable degradation classification loss on the returned latent')
parser.add_argument('--cls_loss_weight', type=float, default=0.1,
                    help='weight of the degradation classification loss')
parser.add_argument('--num_degradation_classes', type=int, default=5,
                    help='number of degradation classes (de_id: noise15=0, noise25=1, noise50=2, rain=3, haze=4)')


# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/train/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/train/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/OTS_BETA/haze/news/',
                    help='where training images of dehazing saves.')
parser.add_argument('--denoise_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/denoising/CBSD68/')
parser.add_argument('--derain_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/deraining/Rain100L/')
parser.add_argument('--dehaze_path', type=str, default='/media/hy/hhh/all-in-one-image-restoration/dataset/dehazing/SOTS/outdoor/')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="restormer",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt/3D/Restormer_classloss",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 2,help = "Number of GPUs to use for training")

options = parser.parse_args()