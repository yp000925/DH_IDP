from __future__ import  print_function
import  matplotlib.pyplot as plt

import os

import numpy as np

from models import *

import torch
import torch.optim

from utils.denoising_utils import *

# dtype = torch.cuda.FlostTensor

imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.

file_path = "/Users/zhangyunping/PycharmProjects/DH_IDP/data/F16_GT.png"

img_pil = crop_image(get_image(file_path,imsize)[0], 32)

img_np = pil_to_np(img_pil)

img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

if PLOT:
    plot_image_grid([img_np, img_noisy_np], 4, 6)

INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

# if fname == 'data/denoising/snail.jpg':
#     num_iter = 2400
#     input_depth = 3
#     figsize = 5
#
#     net = skip(
#         input_depth, 3,
#         num_channels_down=[8, 16, 32, 64, 128],
#         num_channels_up=[8, 16, 32, 64, 128],
#         num_channels_skip=[0, 0, 0, 4, 4],
#         upsample_mode='bilinear',
#         need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
#
#     net = net.type(dtype)

# elif fname == 'data/denoising/F16_GT.png':
dtype = torch.FloatTensor
num_iter = 3000
input_depth = 32
figsize = 4

net = get_net(input_depth, 'skip', pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

# else:
#     assert False

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss()

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
