from os import path
import pdb
import numpy as np
import math
import torch
from opts import opt
import torch.nn.functional as F


class Meter():
    def __init__(self):
        self.reset()

    def add(self, value, n=1):
        self.sum += value * n
        self.n += n
        self.mean = self.sum / self.n

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = -1


# 得到文件后缀名
def file_suffix(filename):
    return path.splitext(filename)[-1]

def clamp_to_0_255(num):
    return max(min(num, 255), 0)

def create_orig_xy_map():
    x = torch.linspace(-1, 1, opt.img_size)
    y = torch.linspace(-1, 1, opt.img_size)
    grid_y, grid_x = torch.meshgrid([x, y])
    grid_x = grid_x.view(1, 1, opt.img_size, opt.img_size)
    grid_y = grid_y.view(1, 1, opt.img_size, opt.img_size)
    orig_xy_map = torch.cat([grid_x, grid_y], 1) # channel stack
    # print (orig_xy_map)
    # pdb.set_trace()
    return orig_xy_map

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_inter_grad(msg):
    def func(x):
        print (msg)
        print (x.norm().item())
    return func


def make_face_region_batch(imgs, face_regions):
    crop_imgs = torch.empty_like(imgs)
    batch_size = imgs.size(0)
    for batch_id in range(batch_size):
        x1 = face_regions[0][0][batch_id]
        y1 = face_regions[0][1][batch_id]
        x2 = face_regions[1][0][batch_id]
        y2 = face_regions[1][1][batch_id]
        # pdb.set_trace()
        # print ('left top: (%d, %d), right bottom: (%d, %d)' % (x1, y1, x2, y2))
        tmp = imgs[batch_id:batch_id+1,:,y1:y2+1,x1:x2+1]
        crop_imgs[batch_id] = F.interpolate(tmp, size=opt.img_size, mode='bilinear', align_corners=True)[0]
    return crop_imgs


if __name__ == '__main__':
    m = Meter()
    pdb.set_trace()