# simple WGAN-GP validator model for synthesis MNIST images
# ref: [https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py]


import torch.nn as nn
import pdb

DIM = 64 # Model dimensionality
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

# ==================Definition Start======================
class MNIST_Generator(nn.Module):
    def __init__(self):
        super(MNIST_Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        # pdb.set_trace()
        # torch.Size([50, 256, 4, 4])
        output = self.block1(output)
        # pdb.set_trace()
        # torch.Size([50, 128, 8, 8])
        #print output.size()
        output = output[:, :, :7, :7]
        # torch.Size([50, 128, 7, 7])
        # pdb.set_trace()
        #print output.size()
        output = self.block2(output)
        # torch.Size([50, 64, 11, 11])

        # pdb.set_trace()
        #print output.size()
        output = self.deconv_out(output)
        # torch.Size([50, 1, 28, 28])

        # pdb.set_trace()
        output = self.sigmoid(output)
        #print output.size()
        # return output.view(-1, OUTPUT_DIM)
        return output.view(-1, 1, 28, 28)

class MNIST_Discriminator(nn.Module):
    def __init__(self):
        super(MNIST_Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

# Cifar10
# ref: [https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py]
class CIFAR10_Generator(nn.Module):
    def __init__(self):
        super(CIFAR10_Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class CIFAR10_Discriminator(nn.Module):
    def __init__(self):
        super(CIFAR10_Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output


# Anime Avatar
# ref: [https://github.com/chenyuntc/pytorch-book/blob/master/chapter7-GAN%E7%94%9F%E6%88%90%E5%8A%A8%E6%BC%AB%E5%A4%B4%E5%83%8F/model.py]


from custom_utils import dotdict
opt = {
    'nz': 128,
    'ngf': 64,
    'ndf': 64,
}

opt = dotdict(opt)

class ANIME_Generator(nn.Module):
    """
    生成器定义
    """
    def __init__(self):
        super(ANIME_Generator, self).__init__()
        ngf = opt.ngf  # 生成器feature map数

        self.main = nn.Sequential(
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def forward(self, input):
        input = input.view(-1, opt.nz, 1, 1)
        return self.main(input)


class ANIME_Discriminator(nn.Module):
    """
    判别器定义
    """
    def __init__(self):
        super(ANIME_Discriminator, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # 输入 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()  # 输出一个数(概率)
        )

    def forward(self, input):
        return self.main(input).view(-1)



# class ANIME_Generator_v2(nn.Module):
#     """
#     生成器定义
#     """
#     def __init__(self):
#         super(ANIME_Generator, self).__init__()
#         ngf = opt.ngf  # 生成器feature map数

#         self.main = nn.Sequential(
#             # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
#             nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # 上一步的输出形状：(ngf*8) x 4 x 4

#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # 上一步的输出形状： (ngf*4) x 8 x 8

#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # 上一步的输出形状： (ngf*2) x 16 x 16

#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # 上一步的输出形状：(ngf) x 32 x 32

#             nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
#             nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
#             # 输出形状：3 x 96 x 96
#         )

#     def forward(self, input):
#         # input = input.view(-1, opt.nz, 1, 1)

#         return self.main(input)