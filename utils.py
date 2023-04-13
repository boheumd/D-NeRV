import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import os
import math
import threading
import numpy as np
import json
import collections
from PIL import Image
from pytorch_msssim import ms_ssim
from tqdm import tqdm

def quantize_per_tensor(t, bit=8, axis=-1):
    if axis == -1:
        t_valid = t!=0
        t_min, t_max =  t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
        elif t.dim() == 5:
            scale = scale[:,None,None,None,None]
            t_min = min_max_tf[:,0,None,None,None,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]      
        elif t.dim() == 5:
            scale = scale[None,:,None,None,None]
            t_min = min_max_tf[None,:,0,None,None,None]    
    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    new_t = t_min + scale * quant_t
    return quant_t, new_t

def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor

def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors

class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def __repr__(self):
        return f"Positional Encoder: pos_b={self.lbase}, pos_l={self.levels}, embed_length={self.embed_length}, to_embed={self.pe_embed}"

    def forward(self, pos):
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase ** (i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            result = torch.stack(pe_list, 1)
            return result

def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str

def adjust_lr(optimizer, cur_epoch, cur_iter, data_size, args, model=None):
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    if cur_epoch < args.warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / args.warmup
    else:
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - args.warmup)/ (args.epochs - args.warmup)) + 1.0)

    lr = args.lr * lr_mult
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr


class WarpKeyframe(nn.Module):
    def __init__(self, height, width, clip_size, device=None):
        super().__init__()
        self.flow_grid = torch.stack(torch.meshgrid(torch.arange(0, height), torch.arange(0, width)), -1).float() #[H, W, 2]
        self.flow_grid = torch.flip(self.flow_grid, (-1,)) # from (y, x) to (x, y)
        self.flow_grid = self.flow_grid.unsqueeze(0) #[H, W, 2] -> [1, H, W, 2]
        self.flow_grid = self.flow_grid.to(device)

        self.height = height
        self.width = width
        self.clip_size = clip_size
        
    def extra_repr(self):
        return 'height={}, width={}, clip_size={}'.format(self.height, self.width, self.clip_size)

    def forward(self, key_frame, output_flow):
        B, C, T, H, W = output_flow.shape
        output_flow = output_flow.permute(0, 2, 3, 4, 1).contiguous().view(B*T, H, W, C) #[B, 2, T, H, W] -> [BT, H, W, 2]
        key_frame = key_frame.permute(0, 2, 1, 3, 4).expand(-1, T, -1, -1, -1).contiguous().view(B*T, -1, H, W) #[B, C, 1, H, W] -> [B, 1, C, H, W] -> [BT, C, H, W]

        next_coords = self.flow_grid.to(output_flow) + output_flow
        next_coords = 2 * next_coords / torch.tensor([[[[W-1, H-1]]]]).to(next_coords) - 1 

        image_warp = F.grid_sample(key_frame, next_coords, padding_mode='border', align_corners=True)

        image_warp = image_warp.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4) # [BT, C, H, W] -> [B, C, T, H, W]
        return image_warp


def split_list(l, n):
    """Yield successive n-sized chunks from l."""
    length = len(l)
    chunk_size = round(length / n)
    for i in range(0, length, chunk_size):
        yield l[i:i + chunk_size]

def psnr(img1, img2):
	mse = torch.mean((img1/255. - img2/255.) ** 2).item()
	if mse < 1.0e-10:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# combine the predicted 256x320 frame patches into 1024x1920 video frame, 
# and then re-evaluate the PSNR/MS-SSIM results of 1024x1920 resolution
@torch.no_grad()
def calculate_metrics_UVG(video_name, video_length, gt_base_dir, pred_base_dir, device):
    h = 1024
    w = 1920
    split_size_h = 256
    split_size_w = 320
    split_num_h = h // split_size_h
    split_num_w = w // split_size_w

    psnr_sum = 0
    msssim_sum = 0
    count = 0
    for frame_index in range(video_length):
        pred_image_list = []
        gt_image_list = []
        for i in range(1, split_num_h * split_num_w + 1):
            pred_image = Image.open(os.path.join(pred_base_dir, "{}-{:02d}".format(video_name, i), 'frame{:06}.png'.format(frame_index + 1))).convert("RGB")
            pred_image_list.append(np.array(pred_image).astype(np.uint8))
            pred_image.close()
            gt_image = Image.open(os.path.join(gt_base_dir, "{}-{:02d}".format(video_name, i), 'frame{:06}.png'.format(frame_index + 1))).convert("RGB")
            gt_image_list.append(np.array(gt_image).astype(np.uint8))
            gt_image.close()
        # combine the split 256x320 frame patches into 1024x1920 full frame
        pred_image = np.stack(pred_image_list, axis=0)
        pred_image = pred_image.reshape(split_num_h, split_num_w, split_size_h, split_size_w, 3)
        pred_image = pred_image.transpose(0, 2, 1, 3, 4).reshape(h, w, 3)
        gt_image = np.stack(gt_image_list, axis=0)
        gt_image = gt_image.reshape(split_num_h, split_num_w, split_size_h, split_size_w, 3)
        gt_image = gt_image.transpose(0, 2, 1, 3, 4).reshape(h, w, 3)

        gt_image_cuda = torch.from_numpy(gt_image).to(torch.float32).to(device)
        pred_image_cuda = torch.from_numpy(pred_image).to(torch.float32).to(device)
        psnr_result = psnr(gt_image_cuda, pred_image_cuda)
        msssim_result = ms_ssim(gt_image_cuda.permute(2, 0, 1).unsqueeze(0), pred_image_cuda.permute(2, 0, 1).unsqueeze(0), data_range=255, size_average=True).item()
        del gt_image_cuda
        del pred_image_cuda
        torch.cuda.empty_cache()
        psnr_sum += psnr_result
        msssim_sum += msssim_result
        print('{}/{}: PSNR:{:.4f} MS-SSIM:{:.4f}'.format(video_name, 'frame{:06}.png'.format(frame_index + 1), psnr_result, msssim_result))
        result_dict['{}/{}'.format(video_name, 'frame{:06}'.format(frame_index + 1))] = {'psnr': psnr_result, 'msssim': msssim_result}

    video_psnr = psnr_sum / video_length
    video_msssim = msssim_sum / video_length

    result_dict['{}'.format(video_name)] = {'psnr': video_psnr, 'msssim': video_msssim, 'clip_size': video_length}

def evaluate_UVG(pred_base_dir, device):
    video_length_list = [["Bosphorus", 600], ["YachtRide", 600], ["HoneyBee", 600], ["ShakeNDry", 300], ["Jockey", 600], ["Beauty", 600], ["ReadySteadyGo", 600]]
    gt_base_dir = 'data/UVG/gt'

    global result_dict
    result_dict = {}

    NUM_THREADS = 4
    splits = list(split_list(video_length_list, NUM_THREADS))

    def target(video_list):
        for video, video_length in tqdm(video_list):
            calculate_metrics_UVG(video, video_length, gt_base_dir, pred_base_dir, device)

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    frame_psnr_total = 0
    frame_msssim_total = 0
    video_size_total = 0
    clip_size_total = 0
    for video_name in result_dict.keys():
        if 'frame' in video_name:
            continue
        video_psnr = result_dict[video_name]['psnr']
        video_msssim = result_dict[video_name]['msssim']
        clip_size = result_dict[video_name]['clip_size']
        frame_psnr_total += video_psnr * clip_size
        frame_msssim_total += video_msssim * clip_size
        clip_size_total += clip_size
    final_psnr = frame_psnr_total / clip_size_total
    final_msssim = frame_msssim_total / clip_size_total
    final_clip_size = clip_size_total

    video_name_list = sorted(result_dict.keys())
    result_dict_sorted = {k: result_dict[k] for k in video_name_list}

    result_dict_sorted['final'] = {'psnr': final_psnr, 'msssim': final_msssim, 'clip_size': final_clip_size}
    print('\nFinal:\n psnr: {:.3f}\n msssim: {:.4f}\n clip_size: {}\n\n'.format(
        final_psnr, final_msssim, final_clip_size))

    result_file_path = os.path.join(pred_base_dir, '../results.json')
    with open(result_file_path, 'w') as fp:
        json.dump(result_dict_sorted, fp, indent=4)
    
    return torch.tensor(final_psnr), torch.tensor(final_msssim)