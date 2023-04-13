import math
import random
import os
import collections
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pytorch_msssim import ms_ssim, ssim
from PIL import Image
from math import pi, sqrt


def loss_fn(pred, target, frame_mask, loss_type='L2'):
    B, C, D, H, W = target.size()
    pred = pred.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
    target = target.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W)
    frame_mask = frame_mask.view(-1)
    pred = pred[frame_mask]
    target = target[frame_mask].detach()
    
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='mean')
    elif loss_type == 'L1':
        loss = torch.mean(torch.abs(pred - target))
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'MSSSIM':
        loss = 1 - ms_ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1': # 0.3 * L2 + 0.7 * SSIM
        loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2': # 0.3 * L1 + 0.7 * SSIM
        loss = 0.3 * torch.mean(torch.abs(pred - target)) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3': # 0.5 * L2 + 0.5 * SSIM
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion4': # 0.5 * L1 + 0.5 * SSIM
        loss = 0.5 * torch.mean(torch.abs(pred - target)) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion5': # 0.7 * L2 + 0.3 * SSIM
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion6': # 0.7 * L1 + 0.3 * SSIM
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion7': # 0.1 * L2 + 0.9 * SSIM
        loss = 0.1 * F.mse_loss(pred, target) + 0.9 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion8': # 0.1 * L1 + 0.9 * SSIM
        loss = 0.1 * torch.mean(torch.abs(pred - target)) + 0.9 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion9': # L2 + SSIM
        loss = 0.9 * F.mse_loss(pred, target) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion10': # L1 + SSIM
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion11': # L2 + SSIM
        loss = 0.95 * F.mse_loss(pred, target) + 0.05 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion12': # L1 + SSIM
        loss = 0.95 * torch.mean(torch.abs(pred - target)) + 0.05 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion13': # L2 + SSIM
        loss = 0.99 * F.mse_loss(pred, target) + 0.01 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion14': # L1 + SSIM
        loss = 0.99 * torch.mean(torch.abs(pred - target)) + 0.01 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion15': # L2 + SSIM
        loss = 0.999 * F.mse_loss(pred, target) + 0.001 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion16': # L1 + SSIM
        loss = 0.999 * torch.mean(torch.abs(pred - target)) + 0.001 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion17': # L2 + SSIM
        loss = F.mse_loss(pred, target) + 0.02 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion18': # L1 + SSIM
        loss = torch.mean(torch.abs(pred - target)) + 0.02 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion19': # L2 + SSIM
        loss = F.mse_loss(pred, target) + 0.03 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion20': # L1 + SSIM
        loss = torch.mean(torch.abs(pred - target)) + 0.03 * (1 - ssim(pred, target, data_range=1, size_average=True))
    return loss


@torch.no_grad()
def psnr_fn(output, target, frame_mask):
    mse = torch.mean((output.detach() - target.detach()) ** 2, dim=(1, 3, 4)) #[B, T]
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    psnr = psnr[frame_mask]
    psnr = torch.mean(psnr)
    return psnr


@torch.no_grad()
def msssim_fn(output, target, frame_mask):
    B, C, D, H, W = output.size()
    output = output.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W).float().detach()
    target = target.permute(0, 2, 1, 3, 4).contiguous().view(B*D, C, H, W).detach()

    if output.size(-2) >= 160:
        msssim = ms_ssim(output, target, data_range=1, size_average=False)
    else:
        msssim = ssim(output, target, data_range=1, size_average=False)

    msssim = msssim[frame_mask.view(-1)]
    msssim = torch.mean(msssim)
    return msssim


