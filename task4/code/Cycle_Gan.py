from glob import glob
import itertools 
import matplotlib.pyplot as plt
import numpy as np
import os
import resnet
import pandas as pd
from PIL import Image
import random
import shutil
import nni
import argparse
import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve
from sklearn import metrics
import time
from tqdm.notebook import tqdm

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, random_split, DataLoader
from nni.utils import merge_parameter

import torchvision.models as models
import torchvision.transforms as transforms

logger = logging.getLogger('Cycle_Gan_AutoML')

# -----| 设置随机数种子，保证训练可以复现 |-----
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# -----| 类继承自Dataset 需要__getitem__,__len__两个函数 |-----
class ImageDataset(Dataset):
    def __init__(self, style_dir, photo_dir, size=(256,256)):
        super().__init__()
        self.style_dir = style_dir
        self.photo_dir = photo_dir
        self.style_idx = dict()
        self.photo_idx = dict()
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        
        for i, img in enumerate(os.listdir(self.style_dir)):
            self.style_idx[i] = img
        for i, img in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = img
    
    #定义了DataLoader中迭代采样的方式    
    def __getitem__(self, idx):
        rand_idx = int(np.random.uniform(0, len(self.style_idx.keys())))
        photo_path = os.path.join(self.photo_dir, self.photo_idx[rand_idx])
        style_path = os.path.join(self.style_dir, self.style_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        style_img = Image.open(style_path)
        style_img = self.transform(style_img)
        return photo_img, style_img
    
    def __len__(self):
        return min(len(self.style_idx.keys()), len(self.photo_idx.keys()))

def Toimg(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(img, mean, std):
        t.mul_(m).add_(s)  
    return img.detach().cpu()

# -----| 上采样网络 | -----
class Upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, padding=1, outpadding=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout(0.5),
            nn.GELU()
        )
    def forward(self, x):
        return self.model(x)

# -----| 下采样网络 |-----
class Subsampled(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, padding=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU()
        )
    def forward(self, x):
        return self.model(x)

# -----| 残差块 |-----
class Resblock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            Subsampled(in_ch, in_ch, 3, 1, 0),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, in_ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(in_ch)
        )
        
    def forward(self, x):
        return x + self.model(x)

# -----| 生成器网络 |-----    
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, g_ch, num_res_blocks=4):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            Subsampled(in_ch, g_ch, 7, 1, 0),
            Subsampled(g_ch, g_ch*2),
            Subsampled(g_ch*2, g_ch*4)
        ]
        for _ in range(num_res_blocks):
            model += [Resblock(g_ch*4)]
        model += [
            Upsampling(g_ch*4, g_ch*2),
            Upsampling(g_ch*2, g_ch),
            nn.ReflectionPad2d(3),
            nn.Conv2d(g_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# -----| 判别器网络 |-----
class Discriminator(nn.Module):
    def __init__(self, in_ch, d_ch, num_layers=4):
        super().__init__()
        model = [
            nn.Conv2d(in_ch, d_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for i in range(1, num_layers):
            in_chs = d_ch * 2**(i-1)
            out_chs = in_chs * 2
            if i == num_layers-1:
                model += [Subsampled(in_chs, out_chs, 4, 1)]
            else:
                model += [Subsampled(in_chs, out_chs, 4, 2)]
        model += [nn.Conv2d(d_ch*8, 1, 4, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# -----| 网络初始化 |-----
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# -----| 网络梯度状态转换 |-----
def update_req_grad(models, requires_grad=True):
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad

# -----| 保存生成图片 |-----
class sample_fake(object):
    def __init__(self, max_imgs=50):
        self.max_imgs = max_imgs
        self.cur_img = 0
        self.imgs = list()
    
    def __call__(self, imgs):
        ret = list()
        for img in imgs:
            if self.cur_img < self.max_imgs:
                self.imgs.append(img)
                ret.append(img)
                self.cur_img += 1
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_imgs)
                    ret.append(self.imgs[idx])
                    self.imgs[idx] = img
                else:
                    ret.append(img)
        return ret

# -----| 定义学习率线性下降 |-----
class lr_sched():
    def __init__(self, decay_epochs=100, total_epochs=200):
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs
        
    def step(self, epoch_num):
        if epoch_num <= self.decay_epochs:
            return 1.0
        else: 
            fract = (epoch_num - self.decay_epochs) / (self.total_epochs - self.decay_epochs)
            return 1.0 - fract

# -----| 初始化 |-----
class AvgStats(object):
    def __init__(self):
        self.resnet()
    
    def resnet(self):
        self.losses = []
        self.its = []
        
    def append(self, loss, it):
        self.losses.append(loss)
        self.its.append(it)

# -----| 建立与训练网络 |-----
class CycleGAN(object):

    # 搭建网络与选择优化器
    def __init__(self, in_ch, out_ch, epochs, device, log_interval, start_lr=2e-4, lmbda=10.0, idt_coef=0.5, g_ch=64, d_ch=64, decay_epoch=0):
        self.eval = resnet.ResNet18()
        self.pool = nn.AvgPool2d(4,4)
        self.eval.load_state_dict(torch.load('./parameter.pkl', map_location='cpu'))
        self.epochs = epochs
        self.decay_epoch = decay_epoch if decay_epoch > 0 else int(self.epochs/2)
        self.lmbda = lmbda
        self.log_interval = log_interval
        self.idt_coef = idt_coef
        self.device = device
        
        self.netG_S2P = Generator(in_ch, out_ch, g_ch)
        self.netG_P2S = Generator(in_ch, out_ch, g_ch)
        self.netD_S = Discriminator(in_ch, d_ch)
        self.netD_P = Discriminator(in_ch, d_ch)
        
        self.init_models()
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adam_gen = torch.optim.Adam(itertools.chain(self.netG_S2P.parameters(), self.netG_P2S.parameters()), lr=start_lr, betas=(0.5, 0.999))
        self.adam_des = torch.optim.Adam(itertools.chain(self.netD_S.parameters(), self.netD_P.parameters()), lr=start_lr, betas=(0.5, 0.999))
        
        self.sample_style = sample_fake()
        self.sample_photo = sample_fake()
        
        gen_lr = lr_sched(self.decay_epoch, self.epochs)
        des_lr = lr_sched(self.decay_epoch, self.epochs)
        self.gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_gen, gen_lr.step)
        self.des_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_des, des_lr.step)
        
        #self.gen_stats = AvgStats()
        #self.des_stats = AvgStats()
    
    # 网络模型与优化器初始化
    def init_models(self):
        self.netG_S2P.apply(weights_init_normal)
        self.netG_P2S.apply(weights_init_normal)
        self.netD_S.apply(weights_init_normal)
        self.netD_P.apply(weights_init_normal)

        self.netG_S2P = self.netG_S2P.to(self.device)
        self.netG_P2S = self.netG_P2S.to(self.device)
        self.netD_S = self.netD_S.to(self.device)
        self.netD_P = self.netD_P.to(self.device)
    
    # 训练过程
    def train(self, imgs):
        possibility__sum = 0.0
        for epoch in range(self.epochs):
            #start_time = time.time()
            possibility_sum = 0.0
            avg_gen_loss = 0.0
            avg_des_loss = 0.0
            #t = tqdm(imgs, leave=False, total=imgs.__len__())
            for i, (photo_real, style_real) in enumerate(imgs):
                photo_img, style_img = photo_real.to(self.device), style_real.to(self.device)
                update_req_grad([self.netD_S, self.netD_P], False)
                self.adam_gen.zero_grad()
                
                # Forward pass through generator
                fake_photo = self.netG_S2P(style_img)
                fake_style = self.netG_P2S(photo_img)
                
                cycl_style = self.netG_P2S(fake_photo)
                cycl_photo = self.netG_S2P(fake_style)
                
                id_style = self.netG_P2S(style_img)
                id_photo = self.netG_S2P(photo_img)
                
                idt_loss_style = self.l1_loss(id_style, style_img) * self.lmbda * self.idt_coef
                idt_loss_photo = self.l1_loss(id_photo, photo_img) * self.lmbda * self.idt_coef
                
                cycle_loss_style = self.l1_loss(cycl_style, style_img) * self.lmbda
                cycle_loss_photo = self.l1_loss(cycl_photo, photo_img) * self.lmbda
                
                style_des = self.netD_S(fake_style)
                photo_des = self.netD_P(fake_photo)
                
                real = torch.ones(style_des.size()).to(self.device)
                
                adv_loss_style = self.mse_loss(style_des, real)
                adv_loss_photo = self.mse_loss(photo_des, real)
                
                total_gen_loss = cycle_loss_style + adv_loss_style + cycle_loss_photo + adv_loss_photo + idt_loss_style + idt_loss_photo
                
                avg_gen_loss += total_gen_loss.item()

                # backward pass
                total_gen_loss.backward()
                self.adam_gen.step()
                
                # Forward pass througgh Descriminator
                update_req_grad([self.netD_S, self.netD_P], True)
                self.adam_des.zero_grad()
                fake_style = self.sample_style([fake_style.cpu().data.numpy()])[0]
                fake_photo = self.sample_photo([fake_photo.cpu().data.numpy()])[0]
                fake_style = torch.tensor(fake_style).to(self.device)
                fake_photo = torch.tensor(fake_photo).to(self.device)
                
                
                style_des_real = self.netD_S(style_img)
                style_des_fake = self.netD_S(fake_style)
                photo_des_real = self.netD_P(photo_img)
                photo_des_fake = self.netD_P(fake_photo)
                
                real = torch.ones(style_des_real.size()).to(self.device)
                fake = torch.ones(style_des_fake.size()).to(self.device)
                
                # Descriminator losses
                # --------------------
                style_des_real_loss = self.mse_loss(style_des_real, real)
                style_des_fake_loss = self.mse_loss(style_des_fake, fake)
                photo_des_real_loss = self.mse_loss(photo_des_real, real)
                photo_des_fake_loss = self.mse_loss(photo_des_fake, fake)
                
                style_des_loss = (style_des_real_loss + style_des_fake_loss) / 2
                photo_des_loss = (photo_des_real_loss + photo_des_fake_loss) / 2
                total_des_loss = style_des_loss + photo_des_loss
                avg_des_loss += total_des_loss.item()

                # Backward
                style_des_loss.backward()
                photo_des_loss.backward()
                self.adam_des.step()
                '''
                # Save network
                save_dict = {
                    'epoch': epoch+1,
                    'netG_S2P': self.netG_S2P.state_dict(),
                    'netG_P2S': self.netG_P2S.state_dict(),
                    'netD_S': self.netD_S.state_dict(),
                    'netD_P': self.netD_P.state_dict(),
                    'optimizer_gen': self.adam_gen.state_dict(),
                    'optimizer_des': self.adam_des.state_dict()
                }
                torch.save(save_dict, 'current.ckpt')
                '''
                avg_gen_loss /= imgs.__len__()
                avg_des_loss /= imgs.__len__()
                #time_req = time.time() - start_time
            
                #self.gen_stats.append(avg_gen_loss, time_req)
                #self.des_stats.append(avg_des_loss, time_req)
                
                if (i+1) % self.log_interval == 0:
                    logger.info("Epoch: (%d) | Generator Loss:%f | Discriminator Loss:%f" % (epoch+1, avg_gen_loss, avg_des_loss))
                    # report intermediate result
                    possibility_sum = F.softmax(F.sigmoid(self.eval(self.pool(fake_style.to('cpu')))), dim = 1)[:, 0:1].sum().item()
                    nni.report_intermediate_result(possibility_sum)

                self.gen_lr_sched.step()
                self.des_lr_sched.step()
        
        # report final result
        nni.report_final_result(possibility_sum)
    
        


# -----| 主函数 |-----
def main(args):

    set_seed(args['seed'])
    use_cuda = args['cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_ds = ImageDataset(args['style_dir'], args['photo_dir'])
    img_dl = DataLoader(img_ds, batch_size=args['batch_size'], pin_memory=True)

    net = CycleGAN(3, 3, args['epochs'], device, args['log_interval'], args['start_lr'], args['lmbda'], args['idt_coef'], args['g_ch'], args['d_ch'])
    net.train(img_dl)  


# -----| 获取参数 |-----
def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cycle_Gan Example')
    parser.add_argument("--style_dir", type=str, default='./data/impressionist', help="style data directory")
    parser.add_argument("--photo_dir", type=str, default='./data/photo_jpg', help="photo data directory")
    parser.add_argument("--batch_size", type=int, default=1, help="number of pictures in each batch (default: 1)")
    parser.add_argument('--epochs', type=int, default=50,  help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=719, help='random seed (default: 719')
    parser.add_argument('--cuda', type=bool, default=False, help='training with gpu or not')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    
    parser.add_argument('--g_ch', type=int, default=16, help="")
    parser.add_argument('--d_ch', type=int, default=16, help="")
    parser.add_argument('--start_lr', type=float, default=0.0001, help="")
    parser.add_argument('--lmbda', type=float, default=0.5, help="")
    parser.add_argument('--idt_coef', type=float, default=0.5, help="")

    
    args, _ = parser.parse_known_args()
    return args

# -----| 程序入口 |-----
if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        #print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise