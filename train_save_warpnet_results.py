from __future__ import division, print_function

import json
import pdb
import random
import time
from os import path

import torch
from torch.utils.data import DataLoader, Dataset

import custom_transforms
import dataset
import models
from custom_criterions import FullSymLoss, MaskedMSELoss, SymLoss, TVLoss
from custom_utils import Meter, create_orig_xy_map, make_dir, weight_init
from opts import opt
from tensorboardX import SummaryWriter
from termcolor import colored
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


class Runner(object):
    def __init__(self):

        # self.writer = SummaryWriter(path.join("tb_logs", opt.exp_name))
    

        self.startup()
        self.prepare_data()
        self.prepare_model()
        self.prepare_optim()
        self.prepare_losses()
        self.load_checkpoint()

    def __del__(self):
        # self.writer.close()
        pass
    
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

            w_gd, grid = self.warpnet(bl, gd)

            pt_l = opt.pt_l_w * self.point_crit(grid, lm_gt, lm_mask)
            tv_l = opt.tv_l_w * self.tv_crit(grid - self.orig_xy_map)
            sym_l = torch.Tensor([0]).to(device)
            if opt.train_sym_dir:
                sym_gt = sb['sym_l'].to(device)
                sym_gd = sb['sym_r'].to(device)
                # sym_l = opt.sym_l_w * self.sym_crit(grid, sym_gd)
                sym_l = opt.sym_l_w * self.sym_crit(grid, sym_gt, sym_gd)
            
            tot_l = pt_l + tv_l + sym_l

            self.warpnet.zero_grad()
            tot_l.backward()
            self.optim.step()
            
            

            self.ms['pt'].add(pt_l.item())
            self.ms['tv'].add(tv_l.item())
            self.ms['sym'].add(sym_l.item())
            self.ms['tot'].add(tot_l.item())

            self.i_batch_tot += 1

            if i_b % opt.print_freq == 0:
                print ('[Train]: %s [%d/%d] (%d/%d)\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tTot Loss=%.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    i_b,
                    self.train_BNPE,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['tot'].mean
                    )
                )

            if self.i_batch_tot % opt.disp_freq == 0:
                self.writer.add_image('train/guide-gt-blur-warp', torch.cat([gd[:opt.disp_img_cnt], gt[:opt.disp_img_cnt], bl[:opt.disp_img_cnt], w_gd[:opt.disp_img_cnt]], 2), self.i_batch_tot)
                self.writer.add_scalar('train/pt_loss', self.ms['pt'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/tv_loss', self.ms['tv'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/sym_loss', self.ms['sym'].mean, self.i_batch_tot)

        print ('*' * 30)
        print ('[Train]: %s [%d/%d]\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tTot Loss=%.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['tot'].mean
                    )
                )
        print ('*' * 30)
        
        


    def test(self, cur_e = 0):
        device = self.device
        for i_b, sb in enumerate(self.test_dl):
            with torch.no_grad():
                gd = sb['guide'].to(device)
                bl = sb['blur'].to(device)
                w_gd, grid = self.warpnet(bl, gd)


                pt_l = torch.Tensor([0]).to(device)
                if opt.test_landmark_dir:
                    lm_mask = sb['lm_mask'].to(device)
                    lm_gt = sb['lm_gt'].to(device) 
                    pt_l = opt.pt_l_w * self.point_crit(grid, lm_gt, lm_mask)
                
                tv_l = opt.tv_l_w * self.tv_crit(grid - self.orig_xy_map)

                sym_l = torch.Tensor([0]).to(device)
                if opt.test_sym_dir:
                    sym_gt = sb['sym_l'].to(device)
                    sym_gd = sb['sym_r'].to(device)
                    # sym_l = opt.sym_l_w * self.sym_crit(grid, sym_gd)
                    sym_l = opt.sym_l_w * self.sym_crit(grid, sym_gt, sym_gd)
            
            tot_l = pt_l + tv_l + sym_l

            self.ms['pt'].add(pt_l.item())
            self.ms['tv'].add(tv_l.item())
            self.ms['sym'].add(sym_l.item())
            self.ms['tot'].add(tot_l.item())

            if i_b % opt.print_freq == 0:
                print ('[Test]: %s [%d/%d] (%d/%d)\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tTot Loss=%.12f' % (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    i_b,
                    self.test_BNPE,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['tot'].mean
                    )
                )
        
        print ('=' * 30)
        print ('[Test]: %s [%d/%d]\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tTot Loss=%.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['tot'].mean
                    )
                )
        print ('=' * 30)
        



    def prepare_losses(self):
        ms = {}
        ms['sym'] = Meter()
        ms['pt'] = Meter()
        ms['tv'] = Meter()
        ms['tot'] = Meter()
        self.ms = ms
        
        if opt.train_sym_dir:
            # self.sym_crit = SymLoss(opt.C)
            self.sym_crit = FullSymLoss(opt.C)
        self.point_crit = MaskedMSELoss()
        self.tv_crit = TVLoss()

        


    def load_checkpoint(self):
        if not opt.load_checkpoint:
            return
        ckpt = torch.load(opt.load_checkpoint)
        self.warpnet.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optim'])
        self.last_epoch = ckpt['epoch']
        self.i_batch_tot = ckpt['i_batch_tot']
        print ('Load ckpt from %s' % opt.load_checkpoint)
        # print ('Cont Train from Epoch %2d' % (self.last_epoch + 1))

        if not opt.load_checkpoint_B:
            return
        ckpt_B = torch.load(opt.load_checkpoint_B)
        self.warpnet_B.load_state_dict(ckpt_B['model'])
        print ('Load ckpt B from %s' % opt.load_checkpoint_B)

        if not opt.load_checkpoint_C:
            return
        ckpt_C = torch.load(opt.load_checkpoint_C)
        self.warpnet_C.load_state_dict(ckpt_C['model'])
        print ('Load ckpt C from %s' % opt.load_checkpoint_C)
        



    def save_checkpoint(self, cur_e = 0):
        ckpt_file = path.join(opt.checkpoint_dir, 'ckpt_%03d.pt' % (cur_e + 1))

        print ('Save Model to %s ... ' % ckpt_file)
        torch.save({
            'epoch': cur_e,
            'i_batch_tot': self.i_batch_tot,
            'model': self.warpnet.state_dict(),
            'optim': self.optim.state_dict(),
        }, ckpt_file)

    def change_model_mode(self, train = True):
        if train:
            self.warpnet.train()
        else:
            self.warpnet.eval()

    def prepare_optim(self):
        betas = (opt.beta1, 0.999)
        self.optim = torch.optim.Adam(self.warpnet.parameters(), lr = opt.lr, betas = betas)

    def prepare_model(self):
        self.warpnet = models.GFRNet_warpnet()
        self.warpnet.to(self.device)
        self.warpnet.apply(weight_init)

        self.warpnet_B = models.GFRNet_warpnet()
        self.warpnet_B.to(self.device)
        self.warpnet_B.apply(weight_init)

        self.warpnet_C = models.GFRNet_warpnet()
        self.warpnet_C.to(self.device)
        self.warpnet_C.apply(weight_init)


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

        if opt.load_sbt_dir:
            self.load_dataset = dataset.LoadFaceDataset(opt.load_sbt_dir, load_tsfm_c)
            self.load_dl = DataLoader(self.load_dataset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers)
            self.load_BNPE = len(self.load_dl)
        else:
            self.load_dataset = self.test_dataset
            self.load_dl = self.test_dl
            self.load_BNPE = self.test_BNPE

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
        # self.writer.add_text('Configs', configs, 0)
        opts_json_path = path.join(opt.checkpoint_dir, 'opts.json')
        with open(opts_json_path, 'w') as f:
            print ('Save opts to %s' % opts_json_path)
            f.write(configs)
        # aux vars
        self.last_epoch = -1
        self.i_batch_tot = 0
        self.orig_xy_map = create_orig_xy_map().to(self.device)
        

    def save_warpnet_test_results(self):
        self.change_model_mode(False)
        device = self.device
        make_dir(opt.str_dir)
        for i_b, sb in tqdm(enumerate(self.load_dl)):
            with torch.no_grad():
                gd = sb['guide'].to(device)
                bl = sb['blur'].to(device)
                gt = sb['gt'].to(device)
                fn = list(map(path.basename, sb['img_path']))
                n_fn = [path.join(opt.str_dir, f_name) for f_name in fn]
                w_gd, grid = self.warpnet(bl, gd)
                w_gd_B, grid_B = self.warpnet_B(bl, gd)
                w_gd_C, grid_C = self.warpnet_C(bl, gd)
                bs = gd.size(0)
                for b_id in tqdm(range(bs)):
                    save_image(torch.cat([gd[b_id], gt[b_id], w_gd[b_id], w_gd_B[b_id], w_gd_C[b_id]], 0).view(5, 3, opt.img_size, opt.img_size), n_fn[b_id], padding = 0)

runner = Runner()
# runner.run()
runner.save_warpnet_test_results()
