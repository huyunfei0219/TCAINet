import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=120, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=5e-4, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=25, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default=os.getcwd() + '/VT821/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default=os.getcwd() + '/VT821/T/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default=os.getcwd() + '/VT821/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default=os.getcwd() + '/VT821/RGB/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default=os.getcwd() + '/VT821/T/', help='the test depth images root')
parser.add_argument('--test_gt_root', type=str, default=os.getcwd() + '/VT821/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default=os.getcwd() + '/TLFNet_LFSOD_cpts/', help='the path to save models and logs')

parser.add_argument('--fs_root', type=str, default=os.getcwd() + '/LFSOD_dataset/RGBD_for_train/FS_rgb/', help='the training FS images root')
parser.add_argument('--test_fs_root', type=str, default=os.getcwd() + '/LFSOD_dataset/test_in_train/FS_rgb/', help='the testing FS images root')

opt = parser.parse_args()
