from __future__ import print_function, division
import os
from os import path
import torch
import cv2
import numpy as np
from custom_utils import file_suffix, clamp_to_0_255
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
import random
from math import floor, ceil
import pdb


# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



opt = {
    'img_size': 256
}


opt = dotdict(opt)

img_suffixes = ['.png']
POS_INF = 1e9
NEG_INF = - POS_INF

Point = namedtuple('Point', ['x', 'y'])

# br, gd, img_path
class LoadFaceDataset(Dataset):
    def __init__(self, img_dir = None, transform = None):
        self.img_dir = img_dir
        self.transform = transform
        self._img_list = []

        for filename in os.listdir(self.img_dir):
            full_filename = path.join(self.img_dir, filename)
            if path.isfile(full_filename) and (file_suffix(filename) in img_suffixes):
                self._img_list.append(filename)
    
    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, idx):
        img_filename = path.join(self.img_dir, self._img_list[idx])
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        wd = int(image.shape[1] // 2)
        left = image[:,:wd,:]
        right = image[:,wd:,:]
        sample = {
            'blur': left,
            'guide': right,
            'img_path': img_filename,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

class FaceDataset(Dataset):
    def __init__(self, img_dir = None, landmark_dir = None, sym_dir = None, flip_prob = 0.5, transform = None, test_mode = False):
        assert not (img_dir is None), "img_dir is None!"
        if not test_mode:
            assert not (landmark_dir is None), "train landmark_dir is None!"
        
        self.mode = "test" if test_mode else "train"
        self.img_dir = img_dir
        self.landmark_dir = landmark_dir
        self.sym_dir = sym_dir

        if self.mode == "test":
            flip_prob = -1

        self.flip_prob = flip_prob
        self.flip_flag = False

        self.transform = transform

        self._img_list = []

        for filename in os.listdir(self.img_dir):
            full_filename = path.join(self.img_dir, filename)
            if path.isfile(full_filename) and (file_suffix(filename) in img_suffixes):
                self._img_list.append(filename)
    
    def __len__(self):
        return len(self._img_list)
    
    # 从文件名得到face region信息
    # 左上角(x1, y1)  右下角(x2, y2)
    # 返回landmark或者sym的对应的文件名
    def parse_filename(self, filename):
        file_id, _, x1, x2, y1, y2 = filename.split('_')
        assert _ == 'NewBB'
        x1 = clamp_to_0_255(int(x1) - 1)
        x2 = clamp_to_0_255(int(x2) - 1)
        y1 = clamp_to_0_255(int(y1) - 1)
        y2 = clamp_to_0_255(int(y2.split('.')[0]) - 1)
        if self.flip_flag:
            x1, x2 = opt.img_size - 1 - x2, opt.img_size - 1 - x1
        return file_id + '.png.txt', [Point(x1, y1), Point(x2, y2)]
  
    # [gt, guide]
    def parse_landmark_file(self, filename):
        lm_l = []
        lm_r = []
        # 左上角是坐标原点
        top_y = POS_INF
        bottom_y = NEG_INF
        left_x = POS_INF
        right_x = NEG_INF


        p_ids = [
            list(range(37, 43)),
            list(range(43, 49)),
            list(range(28, 37)),
            list(range(49, 69))
        ]
        # L/R flip
        if self.flip_flag:
            p_ids[0], p_ids[1] = p_ids[1], p_ids[0]

        p_lens = [len(ids) for ids in p_ids]
        mid_xs = [0] * 4
        mid_ys = [0] * 4
        min_xs = [POS_INF] * 4
        max_xs = [NEG_INF] * 4
        min_ys = [POS_INF] * 4
        max_ys = [NEG_INF] * 4
        

        with open(filename, 'r') as f:
            for idx, line in enumerate(f.readlines(), 1):
                x1, y1, x2, y2 = list(map(float, line.split()))
                x1 -= 1
                y1 -= 1

                if self.flip_flag:
                    x1 = opt.img_size - 1 - x1
                    x2 = -x2
                # gt face region
                top_y = min(top_y, y1)
                bottom_y = max(bottom_y, y1)
                left_x = min(left_x, x1)
                right_x = max(right_x, x1)

                for p in range(4):
                    if idx in p_ids[p]:
                        mid_xs[p] += x1 / p_lens[p]
                        mid_ys[p] += y1 / p_lens[p]
                        min_xs[p] = min(min_xs[p], x1)
                        max_xs[p] = max(max_xs[p], x1)
                        min_ys[p] = min(min_ys[p], y1)
                        max_ys[p] = max(max_ys[p], y1)


                lm_l.append((x1, y1))
                lm_r.append((x2, y2))

        face_region_x1 = clamp_to_0_255(round(left_x))
        face_region_y1 = clamp_to_0_255(round(top_y))
        face_region_x2 = clamp_to_0_255(round(right_x))
        face_region_y2 = clamp_to_0_255(round(bottom_y))

        part_pos = []
        part_expand_mult = [1.2, 1.2, 1.2, 1.2]
        for p in range(4):
            part_pos.append(
                (
                    clamp_to_0_255(round(mid_xs[p])),
                    clamp_to_0_255(round(mid_ys[p])),
                    round(part_expand_mult[p] * max(abs(max_ys[p] - min_ys[p]), abs(max_xs[p] - min_xs[p])))
                )
            )
        assert len(lm_l) == 68 and len(lm_l) == len(lm_r), "Landmarks length must be 68!"
        return lm_l, lm_r, [Point(face_region_x1, face_region_y1), Point(face_region_x2, face_region_y2)], part_pos

    # sym axis is calc by [-1, 1] coords, and y is up increasing which is opposite to array index direction.
    # sym axis is unit vector
    def parse_sym_file(self, filename):
        with open(filename, 'r') as f:
            x_l, y_l, x_r, y_r = list(map(float, f.read().split()))
        if self.flip_flag:
            x_l = -x_l
            x_r = -x_r
        return (x_l, y_l), (x_r, y_r)
    

    def create_lm_gt_mask(self, lm_l, lm_r):
        lm_gt = np.zeros((2, opt.img_size, opt.img_size), dtype=np.float32)
        lm_mask = np.zeros((1, opt.img_size, opt.img_size), dtype=np.float32)

        for idx in range(68):
            x1, y1 = lm_l[idx]
            x2, y2 = lm_r[idx]
            
            floor_x1 = floor(x1)
            ceil_x1 = ceil(x1)
            floor_y1 = floor(y1)
            ceil_y1 = ceil(y1)

            if ceil_x1 > 255 or ceil_y1 > 255 or floor_x1 < 0 or floor_y1 < 0:
                # print ('img_path %s' % self.img_f)
                # print ('skip landmark %d ... ' % idx)
                # pdb.set_trace()
                continue
            # [0,255]
            # the 1st channel is x plane
            # the 2nd channel is y plane
            lm_gt[0][floor_y1][floor_x1] = x2
            lm_gt[0][floor_y1][ceil_x1] = x2
            lm_gt[0][ceil_y1][floor_x1] = x2
            lm_gt[0][ceil_y1][ceil_x1] = x2

            lm_gt[1][floor_y1][floor_x1] = y2
            lm_gt[1][floor_y1][ceil_x1] = y2
            lm_gt[1][ceil_y1][floor_x1] = y2
            lm_gt[1][ceil_y1][ceil_x1] = y2

            lm_mask[0][floor_y1][floor_x1] = 1
            lm_mask[0][floor_y1][ceil_x1] = 1
            lm_mask[0][ceil_y1][floor_x1] = 1
            lm_mask[0][ceil_y1][ceil_x1] = 1

        return lm_gt, lm_mask
            




    def __getitem__(self, idx):
        if self.flip_prob > 0:
            self.flip_flag = random.random() < self.flip_prob
        
        img_filename = path.join(self.img_dir, self._img_list[idx])

        # self.img_f = img_filename

        # image = io.imread(img_filename)
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pdb.set_trace()
        # (H, W, C)
        wd = int(image.shape[1] // 2)
        left = image[:,:wd,:]
        right = image[:,wd:,:]

        # .copy() see [https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663]
        if self.flip_flag:
            left = np.fliplr(left).copy()
            right = np.fliplr(right).copy()
        

        file_id_name, face_region = self.parse_filename(self._img_list[idx])

       
    
        # if self.sym_dir is None, then do not use sym info
        if self.landmark_dir:
            lm_file = path.join(self.landmark_dir, file_id_name)
            lm_l, lm_r, face_region_calc, part_pos = self.parse_landmark_file(lm_file)
            landmark_left = np.array(lm_l, dtype=np.float32)
            landmark_right = np.array(lm_r, dtype=np.float32)

            lm_gt, lm_mask = self.create_lm_gt_mask(landmark_left, landmark_right)


        # if self.sym_dir is None, then do not use sym info
        if self.sym_dir:
            sym_file = path.join(self.sym_dir, file_id_name)
            sym_l, sym_r = self.parse_sym_file(sym_file)
            sym_l, sym_r = np.array(sym_l, dtype=np.float32), np.array(sym_r, dtype=np.float32)
    
        sample = {
            'gt': left,
            'blur': left,
            'guide': right,

            'face_region': face_region,
            'img_path': img_filename,
        }

        if self.landmark_dir:
            sample['part_pos'] = np.array(part_pos, dtype=np.int32)
            sample['face_region_calc'] = face_region_calc
            sample['lm_gt'] = lm_gt
            sample['lm_mask'] = lm_mask
            sample['lm_l'] = landmark_left
            sample['lm_r'] = landmark_right
        
        if self.sym_dir:
            sample['sym_l'] = sym_l
            sample['sym_r'] = sym_r
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def test():
    img_dir = './DataSets/Original/Train'
    landmark_dir = 'DataSets/Original/Landmark'
    sym_dir = None
    # sym_dir = 'DataSets/Original/Sym_bz'
    face_dataset = FaceDataset(img_dir, landmark_dir, sym_dir, -1, None, False)

    # face_dataset = FaceDataset(img_dir, test_mode=True)
    
    print ('Dataset size:', len(face_dataset))

    idx = 1224

    sample = face_dataset[idx]
    face_region = sample['face_region_calc']

    part_pos = sample['part_pos']
    # face_region_calc = sample['face_region_calc']
    print (face_region)
    print (part_pos)

    # pdb.set_trace()
    # print (face_region_calc)

    p1, p2 = face_region
    w = p2.x - p1.x
    h = p2.y - p1.y



    fig, ax = plt.subplots(1,1)
    # 左上角坐标 (宽,高)
    # 坐标原点 左上角
    rect = patches.Rectangle((p1.x,p1.y),w,h,linewidth=1,edgecolor='green',facecolor='none')
    ax.add_patch(rect)
    
    colors = ['r', 'g' , 'w', 'b', 'blue', 'pink', 'purple', 'y']
    for p in range(4):
        L = sample['part_pos'][p]
        rect = patches.Rectangle((L[0] - L[2]/2, L[1] - L[2]/2),L[2],L[2],linewidth=1,edgecolor=random.choice(colors),facecolor='none')
        ax.add_patch(rect)
    # lm_l = sample['lm_l']

    # ax.scatter([50,30],[50,200], s=20, c='blue', marker='o')
    # ax.scatter(lm_l[ : , 0 ], lm_l[ : , 1 ], s=10, c='r', marker='x')
    
    ax.imshow(sample['gt'])

    print (sample['img_path'])
    # print (sample['sym_l'])
    # print (sample['sym_r'])

    plt.savefig('result')

def test_load_dataset():
    load_dataset = LoadFaceDataset("./sbt/sb_7", None)
    idx = 1
    sample = load_dataset[idx]
    print (sample['img_path'])

    fig, ax = plt.subplots(1,1)
    ax.imshow(sample['blur'])
    plt.savefig('load_result')

if __name__ == '__main__':
    # test()
    test_load_dataset()
