import argparse
import os
from os import path
import getpass
import pdb
parser = argparse.ArgumentParser()


# model arch
parser.add_argument('--ngf', type=int, default=64, help='the num of generator(warpNet) 1st conv filters')
parser.add_argument('--output_nc', type=int, default=2, help='the num of generator(warpNet) last conv filters (grid channels)')
parser.add_argument('--img_size', type=int, default=256, help='the image size (current default square)')
parser.add_argument('--output_nc_img', type=int, default=3, help='the num of generator(recNet) output img channels')
parser.add_argument('--part_size', type=int, default=64, help='the part size of eyes, nose and mouth (current default square)')

# train
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate, default=0.0002')
parser.add_argument('--num_workers', type=int, default=8, help="the data loader num workers")
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--max_epoch', type=int, default=150, help='the max num of total training epochs')

parser.add_argument('--exp_name', type=str, default="exp1", help='exp name')

parser.add_argument('--load_warpnet', type=str, default=None, help='the dir of pretrained warpnet ckpt')
parser.add_argument('--load_checkpoint', type=str, default=None, help='the dir to load model checkpoints from for continuing training')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='the dir to save model checkpoints')
parser.add_argument('--save_epoch_freq', type=int, default=20, help='save model every X epochs')


# display
parser.add_argument('--print_freq', type=int, default=100, help='print loss info every X iters')
parser.add_argument('--disp_freq', type=int, default=100, help='refresh the tensorboardX info every X iters')
parser.add_argument('--disp_img_cnt', type=int, default=4, help='the num of displayed images')

# data
parser.add_argument('--train_img_dir', type=str, default=None)
parser.add_argument('--test_img_dir', type=str, default=None)

parser.add_argument('--train_landmark_dir', type=str, default=None)
parser.add_argument('--test_landmark_dir', type=str, default=None)

parser.add_argument('--train_sym_dir', type=str, default=None)
parser.add_argument('--test_sym_dir', type=str, default=None)

parser.add_argument('--flip_prob', type=float, default=0.5, help='the probability to horizontally flip gt & guide for data augmentation')


# loss params

parser.add_argument('--C', type=int, default=10)
parser.add_argument('--pt_l_w', type=float, default=10, help='the point loss weight')
parser.add_argument('--tv_l_w', type=float, default=1, help='the tv loss weight')
parser.add_argument('--sym_l_w', type=float,default=1, help='the sym loss weight')
parser.add_argument('--mse_l_w', type=float, default=0.1, help='the rec mse loss weight, size_average=False')
parser.add_argument('--perp_l_w', type=float, default=0.001, help='the rec perp vgg face loss weight')
parser.add_argument('--gd_l_w', type=float, default=1, help='the global discriminator for G loss weight')
parser.add_argument('--ld_l_w', type=float, default=0.5, help='the local discriminator for G loss weight')

parser.add_argument('--pd_L_l_w', type=float, default=1, help='the part left eye discriminator for G loss weight')
parser.add_argument('--pd_R_l_w', type=float, default=1, help='the part right eye discriminator for G loss weight')
parser.add_argument('--pd_N_l_w', type=float, default=1, help='the part nose discriminator for G loss weight')
parser.add_argument('--pd_M_l_w', type=float, default=1, help='the part mouth discriminator for G loss weight')


# save imgs
# save blurred test images dir
parser.add_argument('--sbt_dir', type=str, default="sbt", help='the base dir to save blurred test images')
# save test results 
parser.add_argument('--str_dir', type=str, default="str", help='the base dir to save test results images')

parser.add_argument('--load_sbt_dir', type=str, default=None, help='the load degradation dataset')


parser.add_argument('--kind', type=str, default="original", help='the degradation kind of test tsfm to save blurred test images')

parser.add_argument('--use_LSGAN', action='store_true', help='whether to use lsgan, remove sigmoid and replace bceloss with mseloss')


parser.add_argument('--hpc_version', action='store_true', help='use on HPC servers')


opt = parser.parse_args()


user_name = getpass.getuser()
if user_name == 'zhangwenqiang':
    opt.hpc_version = True
elif user_name == 'snk':
    opt.hpc_version = False

# if opt.hpc_version:
#     if opt.checkpoint_dir:
#         opt.checkpoint_dir = opt.checkpoint_dir.replace('checkpoints', '/share/data/zwq/GFRNet_pytorch_new/checkpoints')
#     if opt.load_checkpoint:
#         opt.load_checkpoint = opt.load_checkpoint.replace('checkpoints', '/share/data/zwq/GFRNet_pytorch_new/checkpoints')
#     if opt.load_warpnet:
#         opt.load_warpnet = opt.load_warpnet.replace('checkpoints', '/share/data/zwq/GFRNet_pytorch_new/checkpoints')


def make_dir(dir_path):
    if not path.exists(dir_path):
        print ('mkdir', dir_path)
        os.makedirs(dir_path)


opt.checkpoint_dir = path.join(opt.checkpoint_dir, opt.exp_name)
make_dir(opt.checkpoint_dir)

opt.sbt_dir = path.join(opt.sbt_dir, opt.exp_name)
opt.str_dir = path.join(opt.str_dir, opt.exp_name)


