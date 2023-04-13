import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import random
import shutil
import numpy as np
import socket
import logging
from datetime import datetime
from dahuffman import HuffmanCodec
from model import DNeRV, NeRV
from dataset import *
from utils import *
from loss import *
import pdb

def main():
    parser = argparse.ArgumentParser()
    # Model and Dataset configuration
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset')
    parser.add_argument('--model_type', type=str, default='D-NeRV', choices=['NeRV', 'D-NeRV'])
    parser.add_argument('--model_size', type=str, default='S', choices=['XS', 'S', 'M', 'L', 'XL'])
    parser.add_argument('--embed', type=str, default='1.25_240', help='base value/embed length for position encoding')
    parser.add_argument('--spatial_size_h', type=int, default=256)
    parser.add_argument('--spatial_size_w', type=int, default=320)
    parser.add_argument('--keyframe_quality', type=int, default=3, help='keyframe quality, control flag used for keyframe image compression')
    parser.add_argument('--clip_size', type=int, default=8, help='clip_size to sample at a single time')
    parser.add_argument('--fc_hw', type=str, default='4_5', help='out hxw size for mlp')
    parser.add_argument('--fc_dim', type=str, default='100', help='out channel size for mlp')
    parser.add_argument('--expansion', type=float, default=2, help='channel expansion from fc to conv')
    parser.add_argument('--strides', type=int, nargs='+', default=[4, 2, 2, 2, 2], help='strides list')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cos', help='learning rate type, default=cos')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')

    # evaluation parameters
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--quant_model', action='store_true', default=False, help='apply model quantization from torch.float32 to torch.int8')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization, default int8')
    parser.add_argument('--quant_axis', type=int, default=1, help='quantization axis (1 for D-NeRV, 0 for NeRV)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')

    # distribute learning parameters
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str, help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training')

    parser.add_argument('-p', '--print-freq', default=50, type=int,)

    args = parser.parse_args()
    args.warmup = int(args.warmup * args.epochs)
    args.quant_axis = 1 if args.model_type == 'D-NeRV' else 0

    torch.set_printoptions(precision=4) 
    hostname = socket.gethostname()

    # model configs for different architectures, you can change the fc_dim to get different sizes of models
    if args.dataset == 'UVG':
        if args.model_type == 'D-NeRV':
            model_size_dict = {
                'XS': {'fc_dim': 107, 'keyframe_quality': 3},
                'S' : {'fc_dim': 166, 'keyframe_quality': 3},
                'M' : {'fc_dim': 213, 'keyframe_quality': 4},
                'L' : {'fc_dim': 291, 'keyframe_quality': 5},
                'XL': {'fc_dim': 382, 'keyframe_quality': 6},
            }
    elif args.dataset == 'UCF101':
        if args.model_type == 'D-NeRV':
            model_size_dict = {
                'S': {'fc_dim': 198, 'keyframe_quality': 3},
                'M': {'fc_dim': 281, 'keyframe_quality': 3},
                'L': {'fc_dim': 361, 'keyframe_quality': 3},
            }
        elif args.model_type == 'NeRV':
            model_size_dict = {
                'S': {'fc_dim': 465, 'keyframe_quality': -1},
                'M': {'fc_dim': 510, 'keyframe_quality': -1},
                'L': {'fc_dim': 562, 'keyframe_quality': -1},
            }
    args.fc_dim = model_size_dict[args.model_size]['fc_dim']
    args.keyframe_quality = model_size_dict[args.model_size]['keyframe_quality']


    stride_str = '_Strd{}'.format( ','.join([str(x) for x in args.strides]))
    if args.model_type == 'NeRV':
        exp_id = f'{args.dataset}/{args.model_type}/Embed{args.embed}_{args.spatial_size_h}x{args.spatial_size_w}_fc_{args.fc_hw}_{args.fc_dim}_exp{args.expansion}' + \
            f'_f{args.clip_size}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_lr{args.lr}' + \
            f'_{args.loss_type}{stride_str}'
    elif args.model_type == 'D-NeRV':
        exp_id = f'{args.dataset}/{args.model_type}/Embed{args.embed}_{args.spatial_size_h}x{args.spatial_size_w}_fc_{args.fc_hw}_{args.fc_dim}_exp{args.expansion}' + \
            f'_f{args.clip_size}_k{args.keyframe_quality}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_lr{args.lr}' + \
            f'_{args.loss_type}{stride_str}'
    exp_id += '_dist' if args.distributed else ''
    exp_id += '_eval' if args.eval_only else ''
    args.outf = os.path.join('logs', exp_id)
    os.makedirs(args.outf, exist_ok=True)

    port = hash(args.outf) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=3) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = 'cuda:{}'.format(local_rank if local_rank is not None else 0)

    train_best_psnr = torch.tensor(0.0).to(device)
    train_best_msssim = torch.tensor(0.0).to(device)
    val_best_psnr = torch.tensor(0.0).to(device)
    val_best_msssim = torch.tensor(0.0).to(device)
    is_train_best = False

    PE = PositionalEncoding(args.embed)
    args.embed_length = PE.embed_length

    # pre-computed RGB mean and std for the whole video dataset
    if args.dataset == 'UCF101':
        args.num_classes = 101
        args.dataset_mean = [0.3986, 0.3829, 0.3546]
        args.dataset_std = [0.2805, 0.2747, 0.2787]
    elif args.dataset == 'UVG':
        args.num_classes = 7
        args.dataset_mean = [0.4519, 0.4505, 0.4519]
        args.dataset_std = [0.2434, 0.2547, 0.2958]

    if args.model_type == 'NeRV':
        model = NeRV(embed_length=args.embed_length, fc_hw=args.fc_hw, fc_dim=args.fc_dim, expansion=args.expansion, 
                        stride_list=args.strides, lower_width=args.lower_width)
    elif args.model_type == 'D-NeRV':
        model = DNeRV(embed_length=args.embed_length, fc_hw=args.fc_hw, fc_dim=args.fc_dim, expansion=args.expansion, 
                        stride_list=args.strides, lower_width=args.lower_width, 
                        clip_size=args.clip_size, device=device,
                        dataset_mean=args.dataset_mean, dataset_std=args.dataset_std)
    
    ##### get model params and flops #####
    model_params = sum([p.data.nelement() for name, p in model.named_parameters()]) / 1e6
    if local_rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

        print_str = str(model) + '\n' + f'Params: {params}M'
        print(print_str)
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{model_params}M', 'tensorboard'))

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    else:
        model = model.cuda()

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in orig_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(orig_ckt)
        else:
            model.load_state_dict(orig_ckt)
        print("=> loaded checkpoint '{}' (epoch {}) (train_best_psnr {:.2f})".format(args.weight, checkpoint['epoch'], checkpoint['train_best_psnr'].item()))        
        if args.start_epoch < 0:
            args.start_epoch = checkpoint['epoch'] 

    # resume from model_latest
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch'] 
        train_best_psnr = checkpoint['train_best_psnr'].to(device)
        train_best_msssim = checkpoint['train_best_msssim'].to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {}) (train_best_psnr {:.2f})".format(checkpoint_path, checkpoint['epoch'], checkpoint['train_best_psnr'].item()))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    # setup dataloader
    if args.model_type == 'D-NeRV':
        dataset_str = 'Dataset_DNeRV_{}'.format(args.dataset)
    elif args.model_type == 'NeRV':
        dataset_str = 'Dataset_NeRV_{}'.format(args.dataset)
    transform_rgb = transforms.Compose([transforms.ToTensor()])
    transform_keyframe = transforms.Compose([transforms.ToTensor(), transforms.Normalize(args.dataset_mean, args.dataset_std)])
    train_dataset = eval(dataset_str)(args, transform_rgb, transform_keyframe)
    val_dataset = eval(dataset_str)(args, transform_rgb, transform_keyframe)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)

    if args.eval_only:
        print('Evaluation ...')
        val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args, quant_model=args.quant_model, frame_path_list=val_dataset.frame_path_list)
        if args.distributed and args.ngpus_per_node > 1:
            val_psnr = all_reduce(val_psnr.to(local_rank))
            val_msssim = all_reduce(val_msssim.to(local_rank))
        print_str = f'Results for checkpoint: {args.weight}\n\n'
        if args.quant_model:
            print_str += f'[Eval-Quantization] PSNR/MS-SSIM of {args.quant_model_bit} bit with axis {args.quant_axis}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        else:
            print_str += f'[Eval] PSNR/MS-SSIM: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        print(print_str)
        if local_rank in [0, None]:
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
        return

    # Training
    start = datetime.now()
    total_epochs = args.epochs
    for epoch in range(args.start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()

        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []
        for i, (video, norm_idx, keyframe, backward_distance, frame_mask) in enumerate(train_dataloader):
            epoch_ratio = (epoch + float(i) / len(train_dataloader)) / total_epochs

            embed_input = PE(norm_idx)
            video, embed_input, keyframe, backward_distance, frame_mask = video.to(device), embed_input.to(device), \
                                                                        keyframe.to(device), backward_distance.to(device), frame_mask.to(device)

            if args.model_type == 'NeRV':
                output_rgb = model(embed_input)
            elif args.model_type == 'D-NeRV':
                output_rgb = model(embed_input, keyframe=keyframe, backward_distance=backward_distance)

            loss = loss_fn(output_rgb, video, frame_mask, loss_type=args.loss_type)
            lr = adjust_lr(optimizer, epoch, i, len(train_dataloader), args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute psnr and msssim for all the frames
            psnr_list.append(psnr_fn(output_rgb, video, frame_mask))
            msssim_list.append(msssim_fn(output_rgb, video, frame_mask))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                train_psnr = sum(psnr_list) / len(psnr_list)
                train_msssim = sum(msssim_list) / len(msssim_list)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e}, PSNR: {}, MSSSIM: {}, Loss:{}'.format(
                    time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False),
                    RoundTensor(loss, 4, False))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(device)])
            train_msssim = all_reduce([train_msssim.to(device)])

        # add train_psnr to tensorboard
        if local_rank in [0, None]:
            h, w = output_rgb.shape[-2:]
            is_train_best = train_psnr[-1] > train_best_psnr
            train_best_psnr = max(train_psnr[-1], train_best_psnr)
            train_best_msssim = max(train_msssim[-1], train_best_msssim)
            writer.add_scalar(f'Train/PSNR_{h}X{w}', train_psnr[-1], epoch+1)
            writer.add_scalar(f'Train/MSSSIM_{h}X{w}', train_msssim[-1], epoch+1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}', train_best_psnr, epoch+1)
            writer.add_scalar(f'Train/best_MSSSIM_{h}X{w}', train_best_msssim, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            print_str = '\t{}x{}p: current: {:.2f}/{:.2f}\t msssim: {:.4f}/{:.4f}\t'.format(
                h, w, train_psnr[-1].item(), train_best_psnr.item(), train_msssim[-1].item(), train_best_msssim.item())
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')

            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}\n\n\n".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'optimizer': optimizer.state_dict(),   
        }

        # evaluation at the final epoch
        if (epoch + 1) == total_epochs:
            # evaluation without model quantization
            val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(device)])
                val_msssim = all_reduce([val_msssim.to(device)])
            print_str = f'[Eval] PSNR/MS-SSIM: {round(val_psnr[-1].item(),2)}/{round(val_msssim[-1].item(),4)}'
            print(print_str, flush=True)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')

            # evaluation with model quantization
            val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args, quant_model=True)
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(device)])
                val_msssim = all_reduce([val_msssim.to(device)])
            print_str = f'[Eval-Quantization] PSNR/MS-SSIM of {args.quant_model_bit} bit with axis {args.quant_axis}: {round(val_psnr[-1].item(),2)}/{round(val_msssim[-1].item(),4)}'
            print(print_str, flush=True)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n\n')

        if local_rank in [0, None]:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))


@torch.no_grad()
def evaluate(model, val_dataloader, PE, local_rank, args, quant_model=False, frame_path_list=None, mode='train'):
    device = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    visual_dir = f'{args.outf}/visualize'

    ######################### Model Quantization #########################
    if quant_model:
        cur_ckt = model.state_dict()
        quant_weitht_list = []
        for k,v in cur_ckt.items():
            large_tf = (v.dim() in {2,4,5} and 'bias' not in k)
            quant_v, new_v = quantize_per_tensor(v, args.quant_model_bit, args.quant_axis if large_tf else -1)
            valid_quant_v = quant_v[v!=0] # only include non-zero weights
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)    
        encoding_efficiency = avg_bits / args.quant_model_bit
        print_str = f'Entropy encoding efficiency for bit {args.quant_model_bit}: {encoding_efficiency}'
        print(print_str, flush=True)
        if local_rank in [0, None]:
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')       
        model.load_state_dict(cur_ckt)

    ######################### Evaluation #########################
    psnr_list = []
    msssim_list = []
    model.eval()
    for i, (video, norm_idx, keyframe, backward_distance, frame_mask) in enumerate(val_dataloader):
        embed_input = PE(norm_idx)
        video, embed_input, keyframe, backward_distance, frame_mask = video.to(device), embed_input.to(device), \
                                                                    keyframe.to(device), backward_distance.to(device), frame_mask.to(device)

        if args.model_type == 'NeRV':
            output_rgb = model(embed_input)
        elif args.model_type == 'D-NeRV':
            output_rgb = model(embed_input, keyframe=keyframe, backward_distance=backward_distance)
        torch.cuda.synchronize()

        psnr_list.append(psnr_fn(output_rgb, video, frame_mask))
        msssim_list.append(msssim_fn(output_rgb, video, frame_mask))

        # dump predicted frames
        if args.dump_images:
            os.makedirs(visual_dir, exist_ok=True)

            B, C, T, H, W = output_rgb.shape
            for batch_idx in range(B):
                full_idx = i * args.batchSize + batch_idx
                if args.dataset == 'UVG':
                    vid_snippet_name, vid_name, frame_list = frame_path_list[full_idx]
                    os.makedirs(os.path.join(visual_dir, vid_name), exist_ok=True)
                    for t in range(T):
                        save_image(output_rgb[batch_idx, :, t], '{}/{}/{}'.format(visual_dir, vid_name, frame_list[t]))
                else:
                    vid_snippet_name, action_name, vid_name, frame_list = frame_path_list[full_idx]
                    os.makedirs(os.path.join(visual_dir, action_name, vid_name), exist_ok=True)
                    for t in range(T):
                        save_image(output_rgb[batch_idx, :, t], '{}/{}/{}/{}'.format(visual_dir, action_name, vid_name, frame_list[t]))

        val_psnr = sum(psnr_list) / len(psnr_list)
        val_msssim = sum(msssim_list) / len(msssim_list)
        if i % args.print_freq == 0:
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False))
            print(print_str)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')

    # combine the predicted 256x320 frame patches into 1024x1920 video frame, 
    # and re-evaluate the PSNR/MS-SSIM results on 1024x1920 resolution
    if args.dataset == 'UVG' and os.path.exists(visual_dir):
        val_psnr, val_msssim = evaluate_UVG(visual_dir, device)
    return val_psnr, val_msssim

if __name__ == '__main__':
    main()
