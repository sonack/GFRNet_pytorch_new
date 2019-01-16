from stn_module import STN
import torch.nn as nn
from opts import opt
import torch
import ipdb

def warpNet_encoder():
    return  nn.Sequential(
        nn.Conv2d(6, opt.ngf, kernel_size=4, stride=2, padding=1),
        # 128
        nn.LeakyReLU(0.2),
        nn.Conv2d(opt.ngf, opt.ngf * 2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(opt.ngf * 2),
        # 64
        nn.LeakyReLU(0.2),
        nn.Conv2d(opt.ngf * 2, opt.ngf * 4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(opt.ngf * 4),
        # 32
        nn.LeakyReLU(0.2),
        nn.Conv2d(opt.ngf * 4, opt.ngf * 8, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(opt.ngf * 8),
        # 16
        nn.LeakyReLU(0.2),
        nn.Conv2d(opt.ngf * 8, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(opt.ngf * 16),
        # 8
        nn.LeakyReLU(0.2),
        nn.Conv2d(opt.ngf * 16, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(opt.ngf * 16),
        # 4 
        nn.LeakyReLU(0.2),
        nn.Conv2d(opt.ngf * 16, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(opt.ngf * 16),
        # 2
        nn.LeakyReLU(0.2),
        nn.Conv2d(opt.ngf * 16, opt.ngf * 16, kernel_size=4, stride=2, padding=1),
        # nn.BatchNorm2d(opt.ngf * 16),
        # 1
    )

def warpNet_decoder():
    return  nn.Sequential(
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 16, 4, 2, 1),
        nn.BatchNorm2d(opt.ngf * 16),
        # 2
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 16, 4, 2, 1),
        nn.BatchNorm2d(opt.ngf * 16),
        # 4
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 16, 4, 2, 1),
        nn.BatchNorm2d(opt.ngf * 16),
        # 8
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 8, 4, 2, 1),
        nn.BatchNorm2d(opt.ngf * 8),
        # 16
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1),
        nn.BatchNorm2d(opt.ngf * 4),
        # 32
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1),
        nn.BatchNorm2d(opt.ngf * 2),
        # 64
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1),
        nn.BatchNorm2d(opt.ngf),
        # 128
        nn.ReLU(),
        nn.ConvTranspose2d(opt.ngf, opt.output_nc, 4, 2, 1),
        nn.Tanh(),
        # grid [-1,1]
    )

class GFRNet_warpnet(nn.Module):

    def __init__(self):
        super(GFRNet_warpnet, self).__init__()
        # warpNet output flow field
        self.warpNet = nn.Sequential(
            warpNet_encoder(),
            warpNet_decoder()
        )
        self.stn = STN()
    
    def forward(self, blur, guide):
        # pdb.set_trace()
        # ipdb.set_trace()
        pair = torch.cat([blur, guide], 1)  # C = 6
        grid = self.warpNet(pair) # NCHW
        grid_NHWC = grid.permute(0,2,3,1)
        warp_guide = self.stn(guide, grid_NHWC)
        return warp_guide, grid


# recNet
class recNet_encoder_part(nn.Module):
    def __init__(self, ch_in, ch_out, w_bn = True):
        super(recNet_encoder_part, self).__init__()
        modules = [
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
        ]
        if w_bn:
            modules.append(nn.BatchNorm2d(ch_out))
        self.part = nn.Sequential(*modules)

    def forward(self, x):
        return self.part(x)


class recNet_decoder_part(nn.Module):
    def __init__(self, ch_in, ch_out, w_bn = True, w_dp = True):
        super(recNet_decoder_part, self).__init__()
        modules = [
            nn.LeakyReLU(0.2),
        ]

        if opt.deconv_kind == 'resize':
            modules.append(nn.Upsample(scale_factor = 2, mode='bilinear'))
            modules.append(nn.ReflectionPad2d(1))
            modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=0))
        elif opt.deconv_kind == 'deconv':
            modules.append(nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1))      # upsample 2x
        elif opt.deconv_kind == 'subpixel':
            modules.append(nn.Conv2d(ch_in, ch_out * 4, 3, 1, 1))
            modules.append(nn.PixelShuffle(2))
            modules.append(nn.Conv2d(ch_out, ch_out, 1, 1, 0))

        if w_bn:
            modules.append(nn.BatchNorm2d(ch_out))
        if w_dp:
            modules.append(nn.Dropout(0.5))
        self.part = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.part(x)


class GFRNet_recNet(nn.Module):
    def __init__(self):
        super(GFRNet_recNet, self).__init__()
        # make encoder
        # e0 nn.ZeroGrad() ?
        self.encoder = nn.ModuleList()
        enc_ch_multipliers = [1, 2, 4, 8, 8, 8, 8, 8]
        ch_mult = opt.ch_mult
        # e1 ~ e8
        self.encoder.append(nn.Conv2d(3 if opt.minus_WG else 6, opt.ngf * ch_mult, kernel_size=4, stride=2, padding=1))
        for idx in range(2, 9):
            self.encoder.append(recNet_encoder_part(opt.ngf * enc_ch_multipliers[idx-2] * ch_mult, opt.ngf * enc_ch_multipliers[idx-1] * ch_mult, w_bn = (idx != 8)))
        
        # make decoder
        self.decoder = nn.ModuleList()
        dec_ch_in_multipliers = [8, 8*2, 8*2, 8*2, 8*2, 4*2, 2*2]
        dec_ch_out_multipliers = [8, 8, 8, 8, 4, 2, 1]
        # d1 ~ d8
        for idx in range(1, 8):
            w_dp = (idx < 4)
            self.decoder.append(recNet_decoder_part(opt.ngf * dec_ch_in_multipliers[idx-1] * ch_mult, opt.ngf * dec_ch_out_multipliers[idx-1] * ch_mult, w_bn=True, w_dp=w_dp))
        
        if opt.deconv_kind == 'resize':
            self.decoder.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor = 2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(opt.ngf * 2 * ch_mult, opt.output_nc_img, kernel_size=3, stride=1, padding=0)
            ))
        elif opt.deconv_kind == 'deconv':
            self.decoder.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(opt.ngf * 2 * ch_mult, opt.output_nc_img, kernel_size=4, stride=2, padding=1)
            ))
        elif opt.deconv_kind == 'subpixel':
            self.decoder.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(opt.ngf * 2 * ch_mult, opt.output_nc_img * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(opt.output_nc_img, opt.output_nc_img, 1, 1, 0)
            ))

        self.out_act = nn.Sigmoid()
    
    def forward(self, blur, warp_guide = None):
        if opt.minus_WG:
            pair = blur
        else:
            pair = torch.cat([blur, warp_guide], 1)
        # ZeroGrad() or end2end?
        self.encoder_outputs = list(range(8))
        self.encoder_outputs[0] = self.encoder[0](pair)
        for idx in range(1, 8):
            self.encoder_outputs[idx] = self.encoder[idx](self.encoder_outputs[idx-1])
        
        self.decoder_outputs = list(range(8))
        self.decoder_outputs[0] = self.decoder[0](self.encoder_outputs[-1])

        for idx in range(1, 8):
            concat_input = torch.cat([self.decoder_outputs[idx-1], self.encoder_outputs[7-idx]], 1)
            self.decoder_outputs[idx] = self.decoder[idx](concat_input)
        
        self.restored_img = self.out_act(self.decoder_outputs[-1]) # restored image
        return self.restored_img

# generator consists of warpNet and recNet
class GFRNet_generator(nn.Module):
    def __init__(self):
        super(GFRNet_generator, self).__init__()
        if not (opt.minus_W or opt.minus_WG):
            self.warpNet = GFRNet_warpnet()
        self.recNet = GFRNet_recNet()
    
    def forward(self, blur, guide = None):
        if not (opt.minus_W or opt.minus_WG):
            warp_guide, grid = self.warpNet(blur, guide)
            restored_img = self.recNet(blur, warp_guide.detach())
            return warp_guide, grid, restored_img
        else:
            if opt.minus_W: # -W
                restored_img = self.recNet(blur, guide)
            else: # -WG
                restored_img = self.recNet(blur)
            return restored_img


# GAN Global D
class GFRNet_globalDiscriminator(nn.Module):
    def __init__(self, ch_in):
        super(GFRNet_globalDiscriminator, self).__init__()
        w_bn = not opt.noBN_D
        n_layers = 4
        ndf = 64

        modules = []
        modules.append(nn.Conv2d(ch_in, ndf, kernel_size=4, stride=2, padding=1))
        modules.append(nn.LeakyReLU(0.2))

        nf_mult = 1
        nf_mult_prev = 1
        for idx in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**idx, 8)
            modules.append(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 2, 1))
            if w_bn:
                modules.append(nn.BatchNorm2d(ndf*nf_mult))
            modules.append(nn.LeakyReLU(0.2))
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        modules.append(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 2, 1))
        if w_bn:
            modules.append(nn.BatchNorm2d(ndf*nf_mult))
        modules.append(nn.LeakyReLU(0.2))

        # modules.append(nn.Conv2d(ndf*nf_mult, 1, 4, 2))
        modules.append(nn.Conv2d(ndf*nf_mult, ndf*nf_mult, 4, 2))
        modules.append(nn.Conv2d(ndf*nf_mult, 1, 3))
        if not (opt.use_LSGAN or opt.use_WGAN):
            modules.append(nn.Sigmoid())

        self.D = nn.Sequential(*modules)
        # print (self.D)

    def forward(self, x):
        output = self.D(x)
        # pdb.set_trace()
        # return output.view(-1, 1).squeeze(1)
        return output.view(-1)

# GAN Local D
class GFRNet_localDiscriminator(nn.Module):
    def __init__(self, ch_in):
        super(GFRNet_localDiscriminator, self).__init__()
        w_bn = not opt.noBN_D
        n_layers = 4
        ndf = 64

        modules = []
        # modules.append(nn.Upsample((256, 256), mode='bilinear'))
        modules.append(nn.Conv2d(ch_in, ndf, 4, 2, 1))
        modules.append(nn.LeakyReLU(0.2))

        nf_mult = 1
        nf_mult_prev = 1
        for idx in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**idx, 8)
            modules.append(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 2, 1))
            if w_bn:
                modules.append(nn.BatchNorm2d(ndf*nf_mult))
            modules.append(nn.LeakyReLU(0.2))
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        modules.append(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 2, 1))
        if w_bn:
            modules.append(nn.BatchNorm2d(ndf*nf_mult))
        modules.append(nn.LeakyReLU(0.2))

        # modules.append(nn.Conv2d(ndf*nf_mult, 1, 4, 2))
        modules.append(nn.Conv2d(ndf*nf_mult, ndf*nf_mult, 4, 2))
        modules.append(nn.Conv2d(ndf*nf_mult, 1, 3))

        if not (opt.use_LSGAN or opt.use_WGAN):
            modules.append(nn.Sigmoid())

        self.D = nn.Sequential(*modules)
    
    def forward(self, x):
        output = self.D(x)
        # return output.view(-1, 1).squeeze(1)
        return output.view(-1)

# GAN Part D
class GFRNet_partDiscriminator(nn.Module):
    def __init__(self, ch_in):
        super(GFRNet_partDiscriminator, self).__init__()
        w_bn = not opt.noBN_D
        if opt.part_size == 64:
            n_layers = 2 
        elif opt.part_size == 128:
            n_layers = 3
         
        ndf = 64

        modules = []
        # modules.append(nn.Upsample((256, 256), mode='bilinear'))
        modules.append(nn.Conv2d(ch_in, ndf, 4, 2, 1))
        modules.append(nn.LeakyReLU(0.2))

        nf_mult = 1
        nf_mult_prev = 1
        for idx in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**idx, 8)
            modules.append(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 2, 1))
            if w_bn:
                modules.append(nn.BatchNorm2d(ndf*nf_mult))
            modules.append(nn.LeakyReLU(0.2))
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        modules.append(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, 4, 2, 1))
        if w_bn:
            modules.append(nn.BatchNorm2d(ndf*nf_mult))
        modules.append(nn.LeakyReLU(0.2))
        # 8x8  
        # modules.append(nn.Conv2d(ndf*nf_mult, 1, 4, 2))
        modules.append(nn.Conv2d(ndf*nf_mult, ndf*nf_mult, 4, 2))  # 3x3
        modules.append(nn.Conv2d(ndf*nf_mult, 1, 3))
        
        if not (opt.use_LSGAN or opt.use_WGAN):
            modules.append(nn.Sigmoid())

        self.D = nn.Sequential(*modules)
        # print (self.D)
    
    def forward(self, x):
        output = self.D(x)
        # return output.view(-1, 1).squeeze(1)
        return output.view(-1)

if __name__ == '__main__':
    G = GFRNet_generator()
    print (G)



