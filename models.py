from stn_module import STN
import torch.nn as nn
from opts import opt
import torch


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
            nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1),
        ]
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
        # e1 ~ e8
        self.encoder.append(nn.Conv2d(6, opt.ngf, kernel_size=4, stride=2, padding=1))
        for idx in range(2, 9):
            self.encoder.append(recNet_encoder_part(opt.ngf * enc_ch_multipliers[idx-2], opt.ngf * enc_ch_multipliers[idx-1], w_bn = (idx != 8)))
        
        # make decoder
        self.decoder = nn.ModuleList()
        dec_ch_in_multipliers = [8, 8*2, 8*2, 8*2, 8*2, 4*2, 2*2]
        dec_ch_out_multipliers = [8, 8, 8, 8, 4, 2, 1]
        # d1 ~ d8
        for idx in range(1, 8):
            w_dp = (idx < 4)
            self.decoder.append(recNet_decoder_part(opt.ngf * dec_ch_in_multipliers[idx-1], opt.ngf * dec_ch_out_multipliers[idx-1], w_bn=True, w_dp=w_dp))
        self.decoder.append(nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(opt.ngf * 2, opt.output_nc_img, kernel_size=4, stride=2, padding=1)
        ))

        self.out_act = nn.Sigmoid()
    
    def forward(self, blur, warp_guide):
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
        self.warpNet = GFRNet_warpnet()
        self.recNet = GFRNet_recNet()
    
    def forward(self, blur, guide):
        warp_guide, grid = self.warpNet(blur, guide)
        restored_img = self.recNet(blur, warp_guide.detach())
        return warp_guide, grid, restored_img

if __name__ == '__main__':
    G = GFRNet_generator()
    print (G)



