# uncond local GAN

from __future__ import print_function, division
from opts import opt
from tensorboardX import SummaryWriter
import json
from termcolor import colored
import torch
import torch.nn as nn
import custom_transforms
import dataset
from torch.utils.data import Dataset, DataLoader
import models
from custom_utils import weight_init, create_orig_xy_map, Meter, make_face_region_batch

from custom_criterions import MaskedMSELoss, TVLoss, SymLoss, VggFaceLoss
import random
from os import path
from torchvision import transforms
import pdb
import time



real_label = 1
fake_label = 0


class Runner(object):
    def __init__(self):

        self.writer = SummaryWriter(path.join("tb_logs", opt.exp_name))
    
        self.startup()
        self.prepare_data()
        self.prepare_model()
        self.prepare_optim()
        self.prepare_losses()
        self.load_checkpoint()

    def __del__(self):
        self.writer.close()
    

    def prepare_gan_pair_data(self, d, kind = 'global3'):
        if kind == 'global3':
            real = d['gt']
            fake = d['res']
        elif kind == 'local3':
            real = make_face_region_batch(d['gt'], d['f_r'])
            fake = make_face_region_batch(d['res'], d['f_r'])

        # pdb.set_trace()
        return real, fake

    def run(self):
        for e in range(self.last_epoch + 1, opt.max_epoch):
            self.change_model_mode(True)
            self.reset_ms()
            self.train_one_epoch(e)

            self.change_model_mode(False)
            self.reset_ms()
            self.test(e)
            
            if (e + 1) % opt.save_epoch_freq == 0:
                self.save_checkpoint(e)
            print ()


            
    def reset_ms(self):
        for m in self.ms.values():
            m.reset()


    # one epoch train
    def train_one_epoch(self, cur_e = 0):
        device = self.device
        for i_b, sb in enumerate(self.train_dl):
            # if i_b > 100:
            #     break
            gd = sb['guide'].to(device)
            bl = sb['blur'].to(device)
            gt = sb['gt'].to(device)
            lm_mask = sb['lm_mask'].to(device)
            lm_gt = sb['lm_gt'].to(device)
            f_r = sb['face_region_calc']


            local_gt = make_face_region_batch(gt, f_r)

            w_gd, grid, res = self.G(bl, gd)

            pt_l = opt.pt_l_w * self.point_crit(grid, lm_gt, lm_mask)
            tv_l = opt.tv_l_w * self.tv_crit(grid - self.orig_xy_map)
            sym_l = torch.Tensor([0]).to(device)
            if opt.train_sym_dir:
                sym_gd = sb['sym_r'].to(device)
                sym_l = opt.sym_l_w * self.sym_crit(grid, sym_gd)
            
            flow_l = pt_l + tv_l + sym_l

            mse_l = opt.mse_l_w * self.mse_crit(res, gt)
            perp_l = opt.perp_l_w * self.perp_crit(res, gt)

            rec_l = perp_l + mse_l

            # gan loss
            ## Global GAN
            d = {
                'gt': gt,
                'res': res,
                'f_r': f_r,
            }

            batch_size = bl.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            real, fake = self.prepare_gan_pair_data(d)
            
            output = self.GD(fake)
            errGD_G = self.GD_crit(output, label)
            GD_G_l = opt.gd_l_w * errGD_G

            ## Local GAN
            local_real, local_fake = self.prepare_gan_pair_data(d, 'local3')
            output = self.LD(local_fake)
            errLD_G = self.LD_crit(output, label)
            LD_G_l = opt.ld_l_w * errLD_G
            

            # adv_l = LD_G_l
            adv_l = GD_G_l + LD_G_l

            tot_l = flow_l + rec_l + adv_l
            # tot_l = adv_l

            self.G.zero_grad()
            tot_l.backward()
            self.optim.step()
            

            # update D

            ## GD
            self.GD.zero_grad()
            output = self.GD(real)
            label.fill_(real_label)
            errGD_D_real = self.GD_crit(output, label)
            errGD_D_real.backward()
            output = self.GD(fake.detach())
            label.fill_(fake_label)
            errGD_D_fake = self.GD_crit(output, label)
            errGD_D_fake.backward()
            errGD_D = (errGD_D_real + errGD_D_fake) / 2
            self.optimGD.step()
            
            ## LD
            self.LD.zero_grad()
            output = self.LD(local_real)
            label.fill_(real_label)
            errLD_D_real = self.LD_crit(output, label)
            errLD_D_real.backward()
            output = self.LD(local_fake.detach())
            label.fill_(fake_label)
            errLD_D_fake = self.LD_crit(output, label)
            errLD_D_fake.backward()
            errLD_D = (errLD_D_real + errLD_D_fake) / 2
            self.optimLD.step()

            # logging and printing

            self.ms['pt'].add(pt_l.item())
            self.ms['tv'].add(tv_l.item())
            self.ms['sym'].add(sym_l.item())
            self.ms['tot'].add(tot_l.item())
            self.ms['mse'].add(mse_l.item())
            self.ms['perp'].add(perp_l.item())
            self.ms['GD_G'].add(GD_G_l.item())
            self.ms['GD_D'].add(errGD_D.item())
            self.ms['LD_G'].add(LD_G_l.item())
            self.ms['LD_D'].add(errLD_D.item())

            self.i_batch_tot += 1

            if i_b % opt.print_freq == 0:
                print ('[Train]: %s [%d/%d] (%d/%d)\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tMse Loss=%.12f\tPerp Loss=%.12f\tGD Loss: [%.12f/%.12f]\tLD Loss: [%.12f/%.12f]\tTot Loss=%.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    i_b,
                    self.train_BNPE,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['mse'].mean,
                    self.ms['perp'].mean,
                    self.ms['GD_G'].mean,
                    self.ms['GD_D'].mean,
                    self.ms['LD_G'].mean,
                    self.ms['LD_D'].mean,
                    self.ms['tot'].mean,
                    )
                )

            if self.i_batch_tot % opt.disp_freq == 0:
                self.writer.add_image('train/guide-gt-blur-warp-res-local', torch.cat([gd[:opt.disp_img_cnt], gt[:opt.disp_img_cnt], bl[:opt.disp_img_cnt], w_gd[:opt.disp_img_cnt], res[:opt.disp_img_cnt], local_gt[:opt.disp_img_cnt]], 2), self.i_batch_tot)
                
                self.writer.add_scalar('train/mse_loss', self.ms['mse'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/perp_loss', self.ms['perp'].mean, self.i_batch_tot)

                self.writer.add_scalar('train/GD/G', self.ms['GD_G'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/GD/D', self.ms['GD_D'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/LD/G', self.ms['LD_G'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/LD/D', self.ms['LD_D'].mean, self.i_batch_tot)

                self.writer.add_scalar('train/pt_loss', self.ms['pt'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/tv_loss', self.ms['tv'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/sym_loss', self.ms['sym'].mean, self.i_batch_tot)
                


               
        # epoch level loss info
        print ('*' * 30)
        print ('[Train]: %s [%d/%d]\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tMse Loss=%.12f\tPerp Loss=%.12f\tGD Loss: [%.12f/%.12f]\tLD Loss: [%.12f/%.12f]\tTot Loss=%.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['mse'].mean,
                    self.ms['perp'].mean,
                    self.ms['GD_G'].mean,
                    self.ms['GD_D'].mean,
                    self.ms['LD_G'].mean,
                    self.ms['LD_D'].mean,
                    self.ms['tot'].mean,
                    )
                )
        print ('*' * 30)

        self.writer.add_scalar('train/epoch/mse_loss', self.ms['mse'].mean, cur_e)
        self.writer.add_scalar('train/epoch/perp_loss', self.ms['perp'].mean, cur_e)
        


    def test(self, cur_e = 0):
        device = self.device
        for i_b, sb in enumerate(self.test_dl):
            with torch.no_grad():
                gd = sb['guide'].to(device)
                bl = sb['blur'].to(device)
                gt = sb['gt'].to(device)

                w_gd, grid, res = self.G(bl, gd)
        
                pt_l = torch.Tensor([0]).to(device)
                if opt.test_landmark_dir:
                    lm_mask = sb['lm_mask'].to(device)
                    lm_gt = sb['lm_gt'].to(device) 
                    pt_l = opt.pt_l_w * self.point_crit(grid, lm_gt, lm_mask)
                
                tv_l = opt.tv_l_w * self.tv_crit(grid - self.orig_xy_map)

                sym_l = torch.Tensor([0]).to(device)
                if opt.test_sym_dir:
                    sym_gd = sb['sym_r'].to(device)
                    sym_l = opt.sym_l_w * self.sym_crit(grid, sym_gd)
            
                flow_l = pt_l + tv_l + sym_l

                mse_l = opt.mse_l_w * self.mse_crit(res, gt)
                perp_l = opt.perp_l_w * self.perp_crit(res, gt)

                # rec_l = mse_l
                # rec_l = perp_l
                rec_l = mse_l + perp_l

                tot_l = flow_l + rec_l


            self.ms['pt'].add(pt_l.item())
            self.ms['tv'].add(tv_l.item())
            self.ms['sym'].add(sym_l.item())
            self.ms['tot'].add(tot_l.item())
            self.ms['mse'].add(mse_l.item())
            self.ms['perp'].add(perp_l.item())

            if i_b % opt.print_freq == 0:
                print ('[Test]: %s [%d/%d] (%d/%d)\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tMse Loss=%.12f\tPerp Loss=%.12f\tTot Loss=%.12f' % (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    i_b,
                    self.test_BNPE,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['mse'].mean,
                    self.ms['perp'].mean,
                    self.ms['tot'].mean
                    )
                )

                self.writer.add_image('test/guide-gt-blur-warp-res', torch.cat([gd[:opt.disp_img_cnt], gt[:opt.disp_img_cnt], bl[:opt.disp_img_cnt], w_gd[:opt.disp_img_cnt], res[:opt.disp_img_cnt]], 2), self.i_batch_tot)

        
        print ('=' * 30)
        print ('[Test]: %s [%d/%d]\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tMse Loss=%.12f\tPerp Loss=%.12f\tTot Loss=%.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['mse'].mean,
                    self.ms['perp'].mean,
                    self.ms['tot'].mean
                    )
                )
        print ('=' * 30)

        self.writer.add_scalar('test/epoch/mse_loss', self.ms['mse'].mean, cur_e)
        self.writer.add_scalar('test/epoch/perp_loss', self.ms['perp'].mean, cur_e)
        



    def prepare_losses(self):
        ms = {}
        keys = ['sym', 'pt', 'tv', 'mse', 'perp', 'tot', 'GD_G', 'GD_D', 'LD_G', 'LD_D']

        for key in keys:
            ms[key] = Meter()

        self.ms = ms
        
        if opt.train_sym_dir:
            self.sym_crit = SymLoss(opt.C)
        self.point_crit = MaskedMSELoss()
        self.tv_crit = TVLoss()
        self.mse_crit = nn.MSELoss(reduction='sum')
        self.perp_crit = VggFaceLoss(3)
        self.perp_crit.to(self.device)


        self.GD_crit = nn.BCELoss()
        self.LD_crit = nn.BCELoss()

        


    def load_checkpoint(self):
        if not (opt.load_checkpoint or opt.load_warpnet):
            return
        if opt.load_checkpoint:
            ckpt = torch.load(opt.load_checkpoint)
            self.G.load_state_dict(ckpt['model'])
            if 'model_GD' in ckpt:
                self.GD.load_state_dict(ckpt['model_GD'])
            if 'model_LD' in ckpt:
                self.LD.load_state_dict(ckpt['model_LD'])
            # self.optim.load_state_dict(ckpt['optim'])
            if 'optim_GD' in ckpt:
                self.optimGD.load_state_dict(ckpt['optim_GD'])
            if 'optim_LD' in ckpt:
                self.optimLD.load_state_dict(ckpt['optim_LD'])
            self.last_epoch = ckpt['epoch']
            self.i_batch_tot = ckpt['i_batch_tot']
            print ('Cont Train from Epoch %2d' % (self.last_epoch + 1))

        if opt.load_warpnet:
            ckpt = torch.load(opt.load_warpnet)
            self.G.warpNet.load_state_dict(ckpt['model'])
            print ('Load Pretrained Warpnet from %s' % (opt.load_warpnet))


    def save_checkpoint(self, cur_e = 0):
        ckpt_file = path.join(opt.checkpoint_dir, 'ckpt_%03d.pt' % (cur_e + 1))

        print ('Save Model to %s ... ' % ckpt_file)
        torch.save({
            'epoch': cur_e,
            'i_batch_tot': self.i_batch_tot,
            'model': self.G.state_dict(),
            'model_GD': self.GD.state_dict(),
            'model_LD': self.LD.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_GD': self.optimGD.state_dict(),
            'optim_LD': self.optimLD.state_dict(),
        }, ckpt_file)

    def change_model_mode(self, train = True):
        if train:
            for m in self.models:
                m.train()
        else:
            for m in self.models:
                m.eval()

    def prepare_optim(self):
        betas = (opt.beta1, 0.999)
        # self.optim = torch.optim.Adam(self.G.parameters(), lr = opt.lr, betas = betas)
        self.optim = torch.optim.Adam(
            [
                { 'params': self.G.warpNet.parameters(), 'lr': opt.lr * 0.001 },
                { 'params': self.G.recNet.parameters() }
            ],
            lr = opt.lr,
            betas = betas
        )
        self.optimGD = torch.optim.Adam(self.GD.parameters(), lr = opt.lr, betas = betas)
        self.optimLD = torch.optim.Adam(self.LD.parameters(), lr = opt.lr, betas = betas)
       

    def prepare_model(self):
        self.G = models.GFRNet_generator()
        self.G.to(self.device)
        self.G.apply(weight_init)

        # 3 
        # 6
        # 9
        self.GD = models.GFRNet_globalDiscriminator(3)
        self.GD.to(self.device)
        self.GD.apply(weight_init)


        self.LD = models.GFRNet_localDiscriminator(3)
        self.LD.to(self.device)
        self.LD.apply(weight_init)

        self.models = [self.G, self.GD, self.LD]


    def prepare_data(self):
        train_degradation_tsfm = custom_transforms.DegradationModel()
        test_degradation_tsfm = custom_transforms.DegradationModel()
        # train_degradation_tsfm = custom_transforms.DegradationModel("train degradation")
        # test_degradation_tsfm = custom_transforms.DegradationModel("test degradation")
        to_tensor_tsfm = custom_transforms.ToTensor()
        train_tsfms = [
            train_degradation_tsfm,
            to_tensor_tsfm
        ]
        test_tsfms = [
            test_degradation_tsfm,
            to_tensor_tsfm
        ]
        train_tsfm_c = transforms.Compose(train_tsfms)
        test_tsfm_c = transforms.Compose(test_tsfms)
        
        self.train_dataset = dataset.FaceDataset(opt.train_img_dir, opt.train_landmark_dir, opt.train_sym_dir, opt.flip_prob, train_tsfm_c, False)
        self.train_dl = DataLoader(self.train_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers)
        self.train_BNPE = len(self.train_dl)

        self.test_dataset = dataset.FaceDataset(opt.test_img_dir, opt.test_landmark_dir, opt.test_sym_dir, -1, test_tsfm_c, True)
        self.test_dl = DataLoader(self.test_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers)
        self.test_BNPE = len(self.test_dl)

    def startup(self):
        # random seed
        if opt.manual_seed is None:
            opt.manual_seed = random.randint(1, 10000)
        print("Random Seed: ", opt.manual_seed)
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)

        # device
        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
        print ('Use device: %s' % self.device)

        # save_configs
        configs = json.dumps(vars(opt), indent=2)
        print (colored(configs, 'green'))
        self.writer.add_text('Configs', configs, 0)
        opts_json_path = path.join(opt.checkpoint_dir, 'opts.json')
        with open(opts_json_path, 'w') as f:
            print ('Save Opts to %s' % opts_json_path)
            f.write(configs)
        # aux vars
        self.last_epoch = -1
        self.i_batch_tot = 0
        self.orig_xy_map = create_orig_xy_map().to(self.device)
        


runner = Runner()
runner.run()

