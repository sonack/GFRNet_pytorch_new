# only part gan  uncond 3
# cond 6 [wg, gt/res]
# local + part
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
from custom_utils import weight_init, create_orig_xy_map, Meter, make_face_region_batch, make_parts_region_batch, print_inter_grad, calc_gradient_penalty, debug_info

from custom_criterions import MaskedMSELoss, TVLoss, SymLoss, VggFaceLoss
import random
from os import path
from torchvision import transforms
import pdb
import time
from tqdm import tqdm


real_label = 1
fake_label = 0

if opt.hpc_version:
    num = 4
    torch.set_num_threads(num)

def noisy_real_label():
    return random.randint(7, 12) / 10

def noisy_fake_label():
    return random.randint(0, 3) / 10


class Runner(object):
    def __init__(self):

        self.writer = SummaryWriter(path.join("tb_logs", opt.exp_name))
    
        self.startup()
        self.prepare_data()
        self.prepare_model()
        self.prepare_optim()
        self.prepare_losses()
        self.load_checkpoint()

        if opt.debug:
            self.G.recNet.encoder[0].weight.register_hook(print_inter_grad("inter grad func"))

    def __del__(self):
        self.writer.close()
    

    def prepare_gan_pair_data(self, d, kind = 'global3'):
        if kind == 'global3':
            real = d['gt']
            fake = d['res']
        elif kind == 'global6':
            real = torch.cat([d['w_gd'], d['gt']], 1)
            fake = torch.cat([d['w_gd'], d['res']], 1)
        elif kind == 'global9':
            real = torch.cat([d['w_gd'], d['gd'], d['gt']], 1)
            fake = torch.cat([d['w_gd'], d['gd'], d['res']], 1)
        
        elif kind == 'local3':
            real = make_face_region_batch(d['gt'], d['f_r'])
            fake = make_face_region_batch(d['res'], d['f_r'])
        elif kind == 'part3':
            real = make_parts_region_batch(d['gt'], d['p_p'])
            fake = make_parts_region_batch(d['res'], d['p_p'])
        elif kind == 'part6':
            # list len=4
            # list[0] Tensor shape torch.Size([16, 3, 64, 64])
            w_gd_parts = make_parts_region_batch(d['w_gd'], d['p_p'])
            gt_parts = make_parts_region_batch(d['gt'], d['p_p'])
            res_parts = make_parts_region_batch(d['res'], d['p_p'])
            real = [torch.cat([w_gd_part, gt_part], 1) for w_gd_part, gt_part in zip(w_gd_parts, gt_parts)]
            fake = [torch.cat([w_gd_part, res_part], 1) for w_gd_part, res_part in zip(w_gd_parts, res_parts)]
            
        elif kind == 'LR':
            gt_parts = make_parts_region_batch(d['gt'], d['p_p'])
            res_parts = make_parts_region_batch(d['res'], d['p_p'])
            real = torch.cat([gt_parts[0], gt_parts[1]], 1)
            fake = torch.cat([res_parts[0], res_parts[1]], 1)

        # pdb.set_trace()
        return real, fake

    def run(self):
        global gen_iterations
        gen_iterations = 0
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


    def train_G(self):
        ############################
        # (2) Update G network
        ###########################

        # to avoid computation
        for netD in self.models[1:]:
            for p in netD.parameters():
                p.requires_grad = False 
        
        # global real, fake, local_real, local_fake, parts_real, parts_fake, LR_real, LR_fake, label
        # gd, bl, gt, lm_mask, lm_gt, f_r, p_p = *self.pack
        # locals().update(self.pack)
        # pdb.set_trace()
        # local_gt = make_face_region_batch(gt, f_r)
        # print (gd.shape)
        # pdb.set_trace()


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


        # grid.register_hook(grid_grad_func)
        # self.G.recNet.encoder[0].weight.register_hook(inter_grad_func)

        if opt.debug:
            res.register_hook(print_inter_grad("rec tensor grad"))


        # gan loss
        ## Global GAN
        output = self.GD(fake)
        if opt.use_WGAN:
            errGD_G = output.mean()
        else:
            errGD_G = self.GD_crit(output, label)
        GD_G_l = opt.gd_l_w * errGD_G

        ## Local GAN
        output = self.LD(local_fake)
        if opt.use_WGAN:
            errLD_G = output.mean()
        else:
            errLD_G = self.LD_crit(output, label)
        LD_G_l = opt.ld_l_w * errLD_G

        ## Part GAN
        errsPD_G = []
        for p in range(4):
            output = self.PD[p](parts_fake[p])
            if opt.use_WGAN:
                errsPD_G.append(output.mean())
            else:
                errsPD_G.append(self.PD_crit[p](output, label))
        PD_G_l = self.parts_l_w[0] * errsPD_G[0] + self.parts_l_w[1] * errsPD_G[1] + self.parts_l_w[2] * errsPD_G[2] + self.parts_l_w[3] * errsPD_G[3]

        ## LR GAN
        output = self.LR(LR_fake)
        if opt.use_WGAN:
            errLR_G = output.mean()
        else:
            errLR_G = self.LR_crit(output, label)
        LR_G_l = opt.lr_l_w * errLR_G

        adv_l = PD_G_l
        # adv_l = GD_G_l + PD_G_l + LD_G_l + LR_G_l
        # adv_l = GD_G_l + LD_G_l
        # adv_l = GD_G_l
        # adv_l = LD_G_l


        # tot_l = flow_l + rec_l + adv_l
        # tot_l = rec_l + adv_l
        # tot_l = rec_l
        tot_l = adv_l

        self.G.zero_grad()
        tot_l.backward()
        self.optim.step()


        # logging
        self.ms['pt'].add(pt_l.item())
        self.ms['tv'].add(tv_l.item())
        self.ms['sym'].add(sym_l.item())
        self.ms['tot'].add(tot_l.item())
        self.ms['mse'].add(mse_l.item())
        self.ms['perp'].add(perp_l.item())
        self.ms['GD_G'].add(GD_G_l.item())
        self.ms['LD_G'].add(LD_G_l.item())
        self.ms['PD_G'].add(PD_G_l.item())
        self.ms['LR_G'].add(LR_G_l.item())

    def train_Ds(self, end_flag): 
        ############################
        # (1) Update D network
        ###########################
        global label
        for netD in self.models[1:]:
            for p in netD.parameters():
                p.requires_grad = True
        
        ## GD
        ### train with real
        self.GD.zero_grad()
        output = self.GD(real)
        # pdb.set_trace()
        if opt.use_WGAN:
            errGD_D_real = output.mean()
        else:
            label = torch.full_like(label, noisy_real_label())
            errGD_D_real = self.GD_crit(output, label)
        # pdb.set_trace()
        errGD_D_real.backward()

        ### train with fake
        output = self.GD(fake.detach())
        if opt.use_WGAN:
            errGD_D_fake = output.mean() * (-1)
        else:
            label = torch.full_like(label, noisy_fake_label())
            errGD_D_fake = self.GD_crit(output, label)
        errGD_D_fake.backward()

        if opt.use_WGAN_GP:
            gp = calc_gradient_penalty(self.GD, real.data, fake.data)
            gp_l = opt.gp_lambda * gp
            gp_l.backward()

        if opt.use_WGAN:
            # 注意这里是+, 因为errGD_D_fake本身就带有了-号
            errGD_D = errGD_D_real + errGD_D_fake
            wasserstein_dis_GD = - errGD_D
            if opt.use_WGAN_GP:
                errGD_D += gp_l
        else:
            errGD_D = (errGD_D_real + errGD_D_fake) / 2
        self.optimGD.step()
        
        ## LD
        self.LD.zero_grad()
        output = self.LD(local_real)
        if opt.use_WGAN:
            errLD_D_real = output.mean()
        else:
            label = torch.full_like(label, noisy_real_label())
            errLD_D_real = self.LD_crit(output, label)
        errLD_D_real.backward()

        output = self.LD(local_fake.detach())
        if opt.use_WGAN:
            errLD_D_fake = output.mean() * (-1)
        else:
            label = torch.full_like(label, noisy_fake_label())
            errLD_D_fake = self.LD_crit(output, label)
        errLD_D_fake.backward()
        if opt.use_WGAN_GP:
            gp = calc_gradient_penalty(self.LD, local_real.data, local_fake.data)
            gp_l = opt.gp_lambda * gp
            gp_l.backward()
        if opt.use_WGAN:
            errLD_D = errLD_D_real + errLD_D_fake
            wasserstein_dis_LD = - errLD_D
            if opt.use_WGAN_GP:
                errLD_D += gp_l
        else:
            errLD_D = (errLD_D_real + errLD_D_fake) / 2
        self.optimLD.step()

        ## PD
        errsPD_D = []
        dissPD_D = []
        for p in range(4):
            PD = self.PD[p]
            optimPD = self.optimPD[p] 
            part_real = parts_real[p]
            part_fake = parts_fake[p]

            if not opt.use_WGAN:
                PD_crit = self.PD_crit[p]

            PD.zero_grad()
            output = PD(part_real)
            if opt.use_WGAN:
                errPD_D_real = output.mean()
            else:
                label = torch.full_like(label, noisy_real_label())
                errPD_D_real = PD_crit(output, label)
            errPD_D_real.backward()
            output = PD(part_fake.detach())
            if opt.use_WGAN:
                errPD_D_fake = output.mean() * (-1)
            else:
                label = torch.full_like(label, noisy_fake_label())
                errPD_D_fake = PD_crit(output, label)
            errPD_D_fake.backward()
            if opt.use_WGAN_GP:
                gp = calc_gradient_penalty(PD, part_real.data, part_fake.data)
                gp_l = opt.gp_lambda * gp
                gp_l.backward()
            if opt.use_WGAN:
                errPD_D = (errPD_D_real + errPD_D_fake)
                dis_PD = - errPD_D
                dissPD_D.append(dis_PD)
                if opt.use_WGAN_GP:
                    errPD_D += gp_l
            else:
                errPD_D = (errPD_D_real + errPD_D_fake) / 2
            
            errsPD_D.append(errPD_D)
            optimPD.step()

        PD_D_l = errsPD_D[0] + errsPD_D[1] + errsPD_D[2] + errsPD_D[3]
        wasserstein_dis_PD = dissPD_D[0] + dissPD_D[1] + dissPD_D[2] + dissPD_D[3]

        ## LR
        self.LR.zero_grad()
        output = self.LR(LR_real)
        if opt.use_WGAN:
            errLR_D_real = output.mean()
        else:
            # label.fill_(real_label)
            label = torch.full_like(label, noisy_real_label())
            errLR_D_real = self.LR_crit(output, label)
        errLR_D_real.backward()
        output = self.LR(LR_fake.detach())
        if opt.use_WGAN:
            errLR_D_fake = output.mean() * (-1)
        else:
            # label.fill_(fake_label)
            label = torch.full_like(label, noisy_fake_label())
            errLR_D_fake = self.LR_crit(output, label)
        errLR_D_fake.backward()
        if opt.use_WGAN_GP:
            gp = calc_gradient_penalty(self.LR, LR_real.data, LR_fake.data)
            gp_l = opt.gp_lambda * gp
            gp_l.backward()
        if opt.use_WGAN:
            errLR_D = (errLR_D_real + errLR_D_fake)
            wasserstein_dis_LR = - errLR_D
            if opt.use_WGAN_GP:
                errLR_D += gp_l
        else:
            errLR_D = (errLR_D_real + errLR_D_fake) / 2
        self.optimLR.step()


        if opt.use_WGAN and not opt.use_WGAN_GP:
            debug_info ('Weight Clipping!')
            for netD in self.models[1:]:
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        # logging
        if end_flag:
            self.ms['GD_D'].add(errGD_D.item())
            self.ms['LD_D'].add(errLD_D.item())
            for i, p in enumerate(['L', 'R', 'N', 'M']):
                self.ms['PD_D_%c' % p].add(errsPD_D[i].item())
            self.ms['PD_D'].add(PD_D_l.item())
            self.ms['LR_D'].add(errLR_D.item())
            self.ms['GD_dis'].add(wasserstein_dis_GD.item())
            self.ms['LD_dis'].add(wasserstein_dis_LD.item())
            self.ms['PD_dis'].add(wasserstein_dis_PD.item())
            self.ms['LR_dis'].add(wasserstein_dis_LR.item())

    def prepare_all_gans_data(self):
        global real, fake, local_real, local_fake, parts_real, parts_fake, LR_real, LR_fake, label, w_gd, grid, res
        w_gd, grid, res = self.G(bl, gd)
        d = {
            'gt': gt,
            'res': res,
            'w_gd': w_gd.detach(),
            'gd': gd,
            'f_r': f_r,
            'p_p': p_p,
        }

        batch_size = bl.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        real, fake = self.prepare_gan_pair_data(d, 'global%d' % opt.GD_cond)
        local_real, local_fake = self.prepare_gan_pair_data(d, 'local3')
        parts_real, parts_fake = self.prepare_gan_pair_data(d, 'part%d' % opt.PD_cond)
        LR_real, LR_fake = self.prepare_gan_pair_data(d, 'LR')

    # one epoch train
    def train_one_epoch(self, cur_e = 0):
        global device, gen_iterations
        device = self.device
        # grid_grad_meter = Meter()
        # grid_grad_func = print_inter_grad("grid grad", grid_grad_meter)

        # inter_grad_meter = Meter()
        # inter_grad_func = print_inter_grad("recNet encoder[0].weight grad", inter_grad_meter)
        
        
        for i_b, sb in enumerate(self.train_dl):
            # if i_b > 100:
            #     break
            global gd, bl, gt, lm_mask, lm_gt, f_r, p_p
            gd = sb['guide'].to(device)
            bl = sb['blur'].to(device)
            gt = sb['gt'].to(device)
            lm_mask = sb['lm_mask'].to(device)
            lm_gt = sb['lm_gt'].to(device)
            f_r = sb['face_region_calc']
            p_p = sb['part_pos']

            # self.pack = {
            #     'gd': gd,
            #     'bl': bl,
            #     'gt': gt,
            #     # lm means landmark
            #     'lm_mask': lm_mask,
            #     'lm_gt': lm_gt,
            #     'f_r': f_r,
            #     'p_p': p_p
            # }
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters

            range_obj = range(Diters)
            if not opt.hpc_version:
                range_obj = tqdm(range_obj)
                pass
            for iter_of_d in range_obj:
            # for iter_of_d in tqdm(range(Diters)):
            # for iter_of_d in range(Diters):
                self.prepare_all_gans_data()
                self.train_Ds(end_flag = (iter_of_d == Diters - 1))




            # every update G one time, i_batch_tot inc 1
            self.train_G()
            gen_iterations += 1
            self.i_batch_tot += 1

            # printing
            if i_b % opt.print_freq == 0:
                # print ('[%d] inter grad tensor grad scale is' % i_b, inter_grad_meter.mean)
                print ('[Train]: %s [%d/%d] (%d/%d) <%d>\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tMse Loss=%.12f\tPerp Loss=%.12f\tGD Loss: [%.12f/%.12f]\tLD Loss: [%.12f/%.12f]\tPD Loss: [%.12f/%.12f]\tLR Loss: [%.12f/%.12f]\tTot Loss=%.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    i_b,
                    self.train_BNPE,
                    gen_iterations,
                    self.ms['pt'].mean,
                    self.ms['tv'].mean,
                    self.ms['sym'].mean,
                    self.ms['mse'].mean,
                    self.ms['perp'].mean,
                    self.ms['GD_G'].mean,
                    self.ms['GD_D'].mean,
                    self.ms['LD_G'].mean,
                    self.ms['LD_D'].mean,
                    self.ms['PD_G'].mean,
                    self.ms['PD_D'].mean,
                    self.ms['LR_G'].mean,
                    self.ms['LR_D'].mean,
                    self.ms['tot'].mean,
                    )
                )

            # displaying
            if self.i_batch_tot % opt.disp_freq == 0:
                self.writer.add_image('train/guide-gt-blur-warp-res-local_gt-local_res', torch.cat([gd[:opt.disp_img_cnt], gt[:opt.disp_img_cnt], bl[:opt.disp_img_cnt], w_gd[:opt.disp_img_cnt], res[:opt.disp_img_cnt], local_real[:opt.disp_img_cnt], local_fake[:opt.disp_img_cnt]], 2), self.i_batch_tot)
                # self.writer.add_image('train/gt/parts/L-R-N-M', torch.cat([parts_real[0][:opt.disp_img_cnt], parts_real[1][:opt.disp_img_cnt], parts_real[2][:opt.disp_img_cnt], parts_real[3][:opt.disp_img_cnt]], 2), self.i_batch_tot)
                
                self.writer.add_scalar('train/mse_loss', self.ms['mse'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/perp_loss', self.ms['perp'].mean, self.i_batch_tot)

                self.writer.add_scalar('train/GD/G', self.ms['GD_G'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/GD/D', self.ms['GD_D'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/LD/G', self.ms['LD_G'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/LD/D', self.ms['LD_D'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/PD/G', self.ms['PD_G'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/PD/D', self.ms['PD_D'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/LR/G', self.ms['LR_G'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/LR/D', self.ms['LR_D'].mean, self.i_batch_tot)

                self.writer.add_scalar('train/pt_loss', self.ms['pt'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/tv_loss', self.ms['tv'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/sym_loss', self.ms['sym'].mean, self.i_batch_tot)

                # wasserstein distance
                if opt.use_WGAN_GP:
                    self.writer.add_scalar('train/wasserstein_dis/GD', self.ms['GD_dis'].mean, self.i_batch_tot)
                    self.writer.add_scalar('train/wasserstein_dis/LD', self.ms['LD_dis'].mean, self.i_batch_tot)
                    self.writer.add_scalar('train/wasserstein_dis/PD', self.ms['PD_dis'].mean, self.i_batch_tot)
                    self.writer.add_scalar('train/wasserstein_dis/LR', self.ms['LR_dis'].mean, self.i_batch_tot)



               

        print ('*' * 30)
        print ('[Train]: %s [%d/%d]\tPt Loss=%.12f\tTV Loss=%.12f\tSym Loss=%.12f\tMse Loss=%.12f\tPerp Loss=%.12f\tGD Loss: [%.12f/%.12f]\tLD Loss: [%.12f/%.12f]\tPD Loss: [%.12f/%.12f]\tLR Loss: [%.12f/%.12f]\tTot Loss=%.12f' % (
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
                    self.ms['PD_G'].mean,
                    self.ms['PD_D'].mean,
                    self.ms['LR_G'].mean,
                    self.ms['LR_D'].mean,
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
                rec_l = perp_l

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
        keys = ['sym', 'pt', 'tv', 'mse', 'perp', 'tot', 'GD_G', 'GD_D', 'LD_G', 'LD_D', 'PD_G', 'PD_D', 'PD_D_L', 'PD_D_R', 'PD_D_N', 'PD_D_M', 'LR_G', 'LR_D', 'GD_dis', 'LD_dis', 'PD_dis', 'LR_dis']

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


        if not opt.use_WGAN:
            self.GD_crit = nn.BCELoss()
            self.LD_crit = nn.BCELoss()

            self.PD_crit = []
            for p in range(4):
                self.PD_crit.append(nn.BCELoss())

            self.LR_crit = nn.BCELoss()
        
    def load_checkpoint(self):
        if not (opt.load_checkpoint or opt.load_warpnet):
            return
        if opt.load_checkpoint:
            ckpt = torch.load(opt.load_checkpoint)
            self.G.load_state_dict(ckpt['model'])
            if 'model_GD' in ckpt:
                # self.GD.load_state_dict(ckpt['model_GD'])
                pass
            if 'model_LD' in ckpt:
                # print ('model_LD!!')
                # self.LD.load_state_dict(ckpt['model_LD'])
                pass
            if 'model_PD_L' in ckpt:
                for i, p in enumerate(['L', 'R', 'N', 'M']):
                    # self.PD[i].load_state_dict(ckpt['model_PD_%c' % p])
                    pass
            
            # self.optim.load_state_dict(ckpt['optim'])
            if 'optim_GD' in ckpt:
                # self.optimGD.load_state_dict(ckpt['optim_GD'])
                pass
            if 'optim_LD' in ckpt:
                # print ('optim_LD!!')
                # self.optimLD.load_state_dict(ckpt['optim_LD'])
                pass
            if 'optim_PD_L' in ckpt:
                for i, p in enumerate(['L', 'R', 'N', 'M']):
                    # self.optimPD[i].load_state_dict(ckpt['optim_PD_%c' % p])
                    pass
            
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
        ckpt_dict = {
            'epoch': cur_e,
            'i_batch_tot': self.i_batch_tot,
            'model': self.G.state_dict(),
            'model_GD': self.GD.state_dict(),
            'model_LD': self.LD.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_GD': self.optimGD.state_dict(),
            'optim_LD': self.optimLD.state_dict(),
        }
        for i, p in enumerate(['L', 'R', 'N', 'M']):
            ckpt_dict['model_PD_%c' % p] = self.PD[i].state_dict()
            ckpt_dict['optim_PD_%c' % p] = self.optimPD[i].state_dict()
        
        torch.save(ckpt_dict, ckpt_file)

    def change_model_mode(self, train = True):
        if train:
            for m in self.models:
                m.train()
        else:
            for m in self.models:
                m.eval()

    def prepare_optim(self):
        if opt.adam:
            # print ('Enable adam!')
            betas = (opt.beta1, 0.999)
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
            self.optimPD = []
            for p in range(4):
                self.optimPD.append(torch.optim.Adam(self.PD[p].parameters(), lr = opt.lr, betas = betas))
            self.optimLR = torch.optim.Adam(self.LR.parameters(), lr = opt.lr, betas = betas)
        else: # RMSProp
            # print ('Enable RMSProp!')
            self.optim = torch.optim.RMSprop(
                [
                    { 'params': self.G.warpNet.parameters(), 'lr': opt.lr * 0.001 },
                    { 'params': self.G.recNet.parameters() }
                ],
                lr = opt.lr,
                # betas = betas
            )
            self.optimGD = torch.optim.RMSprop(self.GD.parameters(), lr = opt.lr)
            self.optimLD = torch.optim.RMSprop(self.LD.parameters(), lr = opt.lr)
            self.optimPD = []
            for p in range(4):
                self.optimPD.append(torch.optim.RMSprop(self.PD[p].parameters(), lr = opt.lr))
            self.optimLR = torch.optim.RMSprop(self.LR.parameters(), lr = opt.lr)
 
    def prepare_model(self):
        device = self.device
        self.G = models.GFRNet_generator()
        self.G.to(device)
        self.G.apply(weight_init)

        # 3 uncond
        # 6 [w_gd, res/gt]
        # 9 [w_gd, gd, res/gt]
        self.GD = models.GFRNet_globalDiscriminator(opt.GD_cond)
        self.GD.to(device)
        self.GD.apply(weight_init)


        self.LD = models.GFRNet_localDiscriminator(3)
        self.LD.to(device)
        self.LD.apply(weight_init)


        # part Ds
        # [L, R, N, M]
        # cond
        # 3 uncond
        # 6 [w_gd, res/gt]
        self.PD = []
        for p in range(4):
            self.PD.append(models.GFRNet_partDiscriminator(opt.PD_cond))
        
        for pd in self.PD:
            pd.to(device)
            pd.apply(weight_init)
        
        # [L, R]
        self.LR = models.GFRNet_partDiscriminator(6)
        self.LR.to(device)
        self.LR.apply(weight_init)

        self.models = [self.G, self.GD, self.LD, *self.PD, self.LR]

    def prepare_data(self):
        train_degradation_tsfm = custom_transforms.DegradationModel(opt.kind, opt.jpeg_last)
        test_degradation_tsfm = custom_transforms.DegradationModel(opt.kind, opt.jpeg_last)
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
        
        self.train_dataset = dataset.FaceDataset(opt.train_img_dir, opt.train_landmark_dir, opt.train_sym_dir, opt.train_mask_dir, opt.face_masks_dir, opt.flip_prob, train_tsfm_c, False)
        self.train_dl = DataLoader(self.train_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers)
        self.train_BNPE = len(self.train_dl)

        self.test_dataset = dataset.FaceDataset(opt.test_img_dir, opt.test_landmark_dir, opt.test_sym_dir, opt.test_mask_dir, opt.face_masks_dir, -1, test_tsfm_c, True)
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
        self.parts_l_w = [opt.pd_L_l_w, opt.pd_R_l_w, opt.pd_N_l_w, opt.pd_M_l_w]
        # if opt.use_WGAN:
        #     self.one = torch.FloatTensor([1]).to(self.device)
        #     self.mone = (one * -1).to(self.device)


        


runner = Runner()
runner.run()

