from torchvision import transforms, utils
import random
import cv2
import numpy as np
from opts import opt
import pdb
from skimage import io

class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    def __call__(self, sample):
        if 'gt' in sample:
            sample['gt'] = self.to_tensor(sample['gt'])
        if 'blur' in sample:
            sample['blur'] = self.to_tensor(sample['blur'])
        if 'guide' in sample:
            sample['guide'] = self.to_tensor(sample['guide'])
        # if 'mask' in sample:
        #     sample['mask'] = self.to_tensor(sample['mask'])
        return sample



# 图像退化方式

class GaussianBlur(object):
    def __init__(self, sigma=3, size=13):
        assert isinstance(sigma, (int, float))
        self.sigma = sigma
        assert isinstance(size, (int, tuple, list))
        if isinstance(size, int):
            self.size = (size, size) # size must be odd
        else:
            assert len(size) == 2, "len(size) of GaussianBlur must be 2!"
            self.size = size
    
    def __call__(self, sample):
        if self.sigma > 0:
            sample['blur'] = cv2.GaussianBlur(sample['blur'], self.size, self.sigma)
        return sample

class DownSampler(object):
    def __init__(self, scale):
        assert isinstance(scale, (int, float))
        self.scale = scale
    
    def __call__(self, sample):
        if self.scale > 1:
            h, w, _ = sample['blur'].shape
            scaled_h, scaled_w = int(h / self.scale), int(w / self.scale)
            # downsample + upsample
            sample['blur'] = cv2.resize(sample['blur'], (scaled_w, scaled_h), interpolation = cv2.INTER_CUBIC)
        return sample

class UpSampler(object): 
    def __init__(self, scale):
        assert isinstance(scale, (int, float))
        self.scale = scale

    def __call__(self, sample):
        if self.scale > 1:
            sample['blur'] = cv2.resize(sample['blur'], (opt.img_size, opt.img_size), interpolation = cv2.INTER_CUBIC)
        return sample

class AWGN(object):
    def __init__(self, level):
        assert isinstance(level, (int, float))
        self.level = level

    def __call__(self, sample):
        if self.level > 0:
            noise = np.random.randn(*sample['blur'].shape) * self.level
            # clip(0,255) 防止负数变为255
            sample['blur'] = (sample['blur'] + noise).clip(0,255).astype(np.uint8) # otherwise would be np.float64

        return sample

# jpeg compressor + decompressor
class JPEGCompressor(object):
    def __init__(self, quality):
        assert isinstance(quality, (int, float))
        self.quality = quality
    
    def __call__(self, sample):
        if self.quality > 0:    # 0 indicating no lossy compression (i.e losslessly compression)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            sample['blur'] = cv2.imdecode(cv2.imencode('.jpg', sample['blur'], encode_param)[1], 1)
        return sample

class DegradationModel(object):

    def degradation_kind(self, kind = 'original'):
        if kind == 'original':
            # self.msg = msg

            # self.gaussianBlur_sigma_list = [1 + x * 0.1 for x in range(21)]
            self.gaussianBlur_sigma_list = [1 + x for x in range(3)]
            
            self.gaussianBlur_sigma_list += [0]
            # self.gaussianBlur_sigma_list += int(len(self.gaussianBlur_sigma_list)) * [0] # 1/2 trigger this degradation
            
            self.downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
            # self.downsample_scale_list += int(len(self.downsample_scale_list)) * [1]
            
            self.awgn_level_list = list(range(1, 8, 1))
            # self.awgn_level_list += int(len(self.awgn_level_list)) * [0]
            
            self.jpeg_quality_list = list(range(10, 41, 1))
            self.jpeg_quality_list += int(len(self.jpeg_quality_list) * 0.33) * [0]
        
        elif kind == 'only_downsample':
            self.gaussianBlur_sigma_list = [0]
            self.downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
            self.awgn_level_list = [0]
            self.jpeg_quality_list = [0]

        elif kind == 'only_4x':
            self.gaussianBlur_sigma_list = [0]
            # self.downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
            self.downsample_scale_list = [4]
            self.awgn_level_list = [0]
            self.jpeg_quality_list = [0]

        elif kind == 'weaker_1':   # 0.5 trigger prob
            self.gaussianBlur_sigma_list = [1 + x for x in range(3)]
            self.gaussianBlur_sigma_list += int(len(self.gaussianBlur_sigma_list)) * [0] # 1/2 trigger this degradation
            
            self.downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
            self.downsample_scale_list += int(len(self.downsample_scale_list)) * [1]
            
            self.awgn_level_list = list(range(1, 8, 1))
            self.awgn_level_list += int(len(self.awgn_level_list)) * [0]
            
            self.jpeg_quality_list = list(range(10, 41, 1))
            self.jpeg_quality_list += int(len(self.jpeg_quality_list)) * [0]

        elif kind == 'weaker_2':    # weaker than weaker_1, jpeg [20,40]
            self.gaussianBlur_sigma_list = [1 + x for x in range(3)]
            self.gaussianBlur_sigma_list += int(len(self.gaussianBlur_sigma_list)) * [0] # 1/2 trigger this degradation
            
            self.downsample_scale_list = [1 + x * 0.1 for x in range(0,71)]
            self.downsample_scale_list += int(len(self.downsample_scale_list)) * [1]
            
            self.awgn_level_list = list(range(1, 8, 1))
            self.awgn_level_list += int(len(self.awgn_level_list)) * [0]
            
            self.jpeg_quality_list = list(range(20, 41, 1))
            self.jpeg_quality_list += int(len(self.jpeg_quality_list)) * [0]


        
    def __init__(self, kind = 'original', msg = None):
       
        self.gaussianBlur_size_list = list(range(3,14,2))

        self.degradation_kind(kind)


        # ops
        self.gaussianBlur = GaussianBlur(random.choice(self.gaussianBlur_sigma_list), random.choice(self.gaussianBlur_size_list))
        self.downSampler = DownSampler(random.choice(self.downsample_scale_list))
        self.upSampler = UpSampler(self.downSampler.scale)
        self.awgn = AWGN(random.choice(self.awgn_level_list))
        self.jpegCompressor = JPEGCompressor(random.choice(self.jpeg_quality_list))
            

    def random_params(self):
        self.gaussianBlur.sigma = random.choice(self.gaussianBlur_sigma_list)
        self.gaussianBlur.size = (random.choice(self.gaussianBlur_size_list),) * 2

        self.downSampler.scale = random.choice(self.downsample_scale_list)
        self.upSampler.scale = self.downSampler.scale
        self.awgn.level = random.choice(self.awgn_level_list)
        self.jpegCompressor.quality = random.choice(self.jpeg_quality_list)


    def __call__(self, sample):
        # print (self.msg)
        self.random_params()
        return self.upSampler(self.jpegCompressor(self.awgn(self.downSampler(self.gaussianBlur(sample)))))


def test_jpeg():
    test_img = './sn.jpg'
    gauss = GaussianBlur(1)
    awgn = AWGN(100)
    jpeg_d = JPEGCompressor(20)
    # img = io.imread(test_img)
    img = cv2.imread(test_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pdb.set_trace()
    img_d = awgn({'blur':img})['blur']
    img_d_bgr = cv2.cvtColor(img_d, cv2.COLOR_RGB2BGR)
    # pdb.set_trace()
    cv2.imwrite('sn_jpeg.png', img_d_bgr)


def test_DegradationModel():
    test_img = './sn.jpg'
    degradationModel = DegradationModel()
    img = cv2.imread(test_img)
    cv2.imwrite("degraded_result.png", degradationModel({'blur':img})['blur'])
    # cv2.imwrite("degraded_result.png", img)

if __name__ == '__main__':
    # test_DegradationModel()
	test_jpeg()
