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
from custom_utils import weight_init, create_orig_xy_map, Meter, make_face_region_batch, make_parts_region_batch, print_inter_grad, calc_gradient_penalty, debug_info, dotdict

from custom_criterions import MaskedMSELoss, TVLoss, SymLoss, VggFaceLoss
import random
from os import path
from torchvision import transforms
import pdb
import time
from tqdm import tqdm
import torchvision.datasets as datasets
import numpy as np
import validator_models as v_models
from torchvision.utils import make_grid
if opt.hpc_version:
    num = 4
    torch.set_num_threads(num)

real_label = 1
fake_label = 0

def noisy_real_label():
    return random.randint(7, 12) / 10

def noisy_fake_label():
    return random.randint(0, 3) / 10

# 单独为 validator gan 设置的超参
config = {
    'nz': 128, # size of the latent z vector
    # 'dataset': 'mnist', # 'mnist', 'cifar10'
    'dataset': 'cifar10',
}

config = dotdict(config)

print (config)

# pdb.set_trace()

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
    

    def run(self):
        global gen_iterations
        gen_iterations = 0
        for e in range(self.last_epoch + 1, opt.max_epoch):
            self.change_model_mode(True)
            self.reset_ms()
            self.train_one_epoch(e)

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
        
        output = self.D(fake)
        if opt.use_WGAN:
            err_G = output.mean()
        else:
            err_G = self.D_crit(output, label)

        G_l = err_G
        adv_l = G_l
        tot_l = adv_l

        self.G.zero_grad()
        tot_l.backward()
        self.optim.step()

        # logging
        self.ms['G'].add(G_l.item())

    def train_Ds(self, end_flag): 
        ############################
        # (1) Update D network
        ###########################
        global label
        for netD in self.models[1:]:
            for p in netD.parameters():
                p.requires_grad = True
        ## D
        ### train with real
        self.D.zero_grad()
        output = self.D(real)
        # pdb.set_trace()
        if opt.use_WGAN:
            err_D_real = output.mean()
        else:
            label = torch.full_like(label, noisy_real_label())
            err_D_real = self.D_crit(output, label)
        # pdb.set_trace()
        err_D_real.backward()

        ### train with fake
        output = self.D(fake.detach())
        if opt.use_WGAN:
            err_D_fake = output.mean() * (-1)
        else:
            label = torch.full_like(label, noisy_fake_label())
            err_D_fake = self.D_crit(output, label)
        err_D_fake.backward()

        if opt.use_WGAN_GP:
            gp = calc_gradient_penalty(self.D, real.data, fake.data)
            # gp = calc_gradient_penalty_mnist(self.D, real.data, fake.data)
            gp_l = opt.gp_lambda * gp
            self.ms['gp'].add(gp_l.item())
            gp_l.backward()

        if opt.use_WGAN:
            # 注意这里是+, 因为errGD_D_fake本身就带有了-号
            err_D = err_D_real + err_D_fake
            wasserstein_dis = - err_D
            if opt.use_WGAN_GP:
                err_D += gp_l
        else:
            err_D = (err_D_real + err_D_fake) / 2
        self.optimD.step()

        if opt.use_WGAN and not opt.use_WGAN_GP:
            debug_info ('Weight Clipping!')
            for netD in self.models[1:]:
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        # logging
        if end_flag:
            self.ms['D'].add(err_D.item())
            self.ms['dis'].add(wasserstein_dis.item())

    def prepare_all_gans_data(self):
        global fake
        fake = self.G(noise)

    # one epoch train
    def train_one_epoch(self, cur_e = 0):
        global device, gen_iterations
        device = self.device
        
        for i_b, sb in enumerate(self.train_dl):
            # if i_b > 100:
            #     break
            # pdb.set_trace()
            global real, noise
            real, _ = sb
            # real = real.view(-1, 28*28).to(device)
            real = real.to(device)

            noise = torch.randn(real.size(0), config.nz).to(device)

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

            if i_b % opt.print_freq == 0:
                print ('[Train]: %s [%d/%d] (%d/%d) <%d>\tGAN Loss: [%.12f/%.12f]\tGP Loss: %.12f\tWasserstein Distance: %.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    i_b,
                    self.train_BNPE,
                    gen_iterations,
                    self.ms['G'].mean,
                    self.ms['D'].mean,
                    self.ms['gp'].mean,
                    self.ms['dis'].mean,
                    )
                )

            # displaying
            if self.i_batch_tot % opt.disp_freq == 0:
                # self.writer.add_image('train/real-fake', torch.cat([real.view(-1, 1, 28, 28)[:opt.disp_img_cnt], fake.view(-1, 1, 28, 28)[:opt.disp_img_cnt]], 2), self.i_batch_tot)
                self.writer.add_image('train/real', make_grid(real, nrow = 10 if config.dataset == 'mnist' else 8, padding = 0, normalize = True), self.i_batch_tot)
                self.writer.add_image('train/fake', make_grid(fake, nrow = 10 if config.dataset == 'mnist' else 8, padding = 0, normalize = True), self.i_batch_tot)

                self.writer.add_scalar('train/G', self.ms['G'].mean, self.i_batch_tot)
                self.writer.add_scalar('train/D', self.ms['D'].mean, self.i_batch_tot)
                if opt.use_WGAN_GP:
                    self.writer.add_scalar('train/gp', self.ms['gp'].mean, self.i_batch_tot)
                    self.writer.add_scalar('train/wasserstein_dis', self.ms['dis'].mean, self.i_batch_tot)


        print ('*' * 30)
        print ('[Train]: %s [%d/%d]\tGAN Loss: [%.12f/%.12f]\tGP Loss: %.12f\tWasserstein Distance: %.12f' % (
                    time.strftime("%m-%d %H:%M:%S", time.localtime()),
                    cur_e,
                    opt.max_epoch,
                    self.ms['G'].mean,
                    self.ms['D'].mean,
                    self.ms['gp'].mean,
                    self.ms['dis'].mean,
                    )
                )
        print ('*' * 30)


        self.writer.add_scalar('train/epoch/G', self.ms['G'].mean, cur_e)
        self.writer.add_scalar('train/epoch/D', self.ms['D'].mean, cur_e)
        if opt.use_WGAN_GP:
            self.writer.add_scalar('train/epoch/gp', self.ms['gp'].mean, cur_e)
            self.writer.add_scalar('train/epoch/wasserstein_dis', self.ms['dis'].mean, cur_e)

        
    def prepare_losses(self):
        ms = {}
        keys = ['G', 'D', 'dis', 'gp']

        for key in keys:
            ms[key] = Meter()

        self.ms = ms
    
        if not opt.use_WGAN:
            self.D_crit = nn.BCELoss()
        
    def load_checkpoint(self):
        # if not (opt.load_checkpoint or opt.load_warpnet):
        if not opt.load_checkpoint:
            return
        if opt.load_checkpoint:
            ckpt = torch.load(opt.load_checkpoint)
            self.G.load_state_dict(ckpt['model'])
            self.D.load_checkpoint(ckpt['model_D'])
            self.optim.load_state_dict(ckpt['optim'])
            self.optimD.load_state_dict(ckpt['optim_D'])

            self.last_epoch = ckpt['epoch']
            self.i_batch_tot = ckpt['i_batch_tot']
            print ('Cont Train from Epoch %2d' % (self.last_epoch + 1))

    def save_checkpoint(self, cur_e = 0):
        ckpt_file = path.join(opt.checkpoint_dir, 'ckpt_%03d.pt' % (cur_e + 1))

        print ('Save Model to %s ... ' % ckpt_file)

        ckpt_dict = {
            'epoch': cur_e,
            'i_batch_tot': self.i_batch_tot,
            'model': self.G.state_dict(),
            'model_D': self.D.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_D': self.optimD.state_dict(),
        }        
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
            self.optim = torch.optim.Adam(self.G.parameters(), lr = opt.lr, betas = betas)
            self.optimD = torch.optim.Adam(self.D.parameters(), lr = opt.lr, betas = betas)
        else: # RMSProp
            # print ('Enable RMSProp!')
            self.optim = torch.optim.RMSprop(self.G.parameters(), lr = opt.lr)
            self.optimD = torch.optim.RMSprop(self.D.parameters(), lr = opt.lr)
 
    def prepare_model(self):
        device = self.device
        if config.dataset == 'mnist':
            self.G = v_models.MNIST_Generator()
        elif config.dataset == 'cifar10':
            self.G = v_models.CIFAR10_Generator()
        self.G.to(device)
        self.G.apply(weight_init)

        if config.dataset == 'mnist':
            self.D = v_models.MNIST_Discriminator()
        elif config.dataset == 'cifar10':
            self.D = v_models.CIFAR10_Discriminator()
        self.D.to(device)
        self.D.apply(weight_init)

        self.models = [self.G, self.D]

    def prepare_data(self):
        if config.dataset == 'mnist':
            trainset = datasets.MNIST(root='./playground/validator_data/mnist', train=True, download=False, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        elif config.dataset == 'cifar10':
            trainset = datasets.CIFAR10(root='./playground/validator_data/cifar10', train=True, download=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

        self.train_dataset = trainset
        self.train_dl = DataLoader(self.train_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers)
        self.train_BNPE = len(self.train_dl)

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

runner = Runner()
runner.run()