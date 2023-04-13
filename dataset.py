import math
import random
import os
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from torch.utils.data import Dataset
from PIL import Image
from math import pi, sqrt
from utils import WarpKeyframe

class Dataset_DNeRV_UVG(Dataset):
    def __init__(self, args, transform_rgb, transform_keyframe=None):
        self.gt_base_dir = './data/UVG/gt'
        self.keyframe_base_dir = './data/UVG/keyframe/q{}'.format(args.keyframe_quality)
        self.transform_rgb = transform_rgb
        self.transform_keyframe = transform_keyframe

        vid_length_dict = collections.OrderedDict()
        with open('./data/UVG/annotation/video_length.json', 'r') as fp:
            vid_length_dict = json.load(fp)

        clip_size = args.clip_size

        vid_dict = collections.OrderedDict()
        self.frame_count_total = 0
        self.frame_path_list = []
        for vid_name, vid_length in vid_length_dict.items():
            # we divide videos into consecutive video_clips
            num_clip = round(math.ceil(vid_length / clip_size))
            # rounded up the vid_length, in case the vid_length is not divided by the clip_size
            vid_length_round = num_clip * clip_size

            for clip_index in range(num_clip):
                # the first frame is the start_keyframe, the first frame for the next consecutive clip is the end_keyframe
                start_keyframe_index = clip_index * clip_size + 1
                end_keyframe_index = min(vid_length, (clip_index + 1) * clip_size + 1)

                vid_clip_name = "{}-{}".format(vid_name, clip_index)
                vid_dict[vid_clip_name] = {}
                vid_dict[vid_clip_name]['vid_name'] = vid_name
                vid_dict[vid_clip_name]['keyframe_path'] = ['frame{:06d}.png'.format(start_keyframe_index), 'frame{:06d}.png'.format(end_keyframe_index)]
                frame_index_list = list(range(clip_index * clip_size + 1, (clip_index + 1) * clip_size + 1))
                # mask out the frame_index which are longer than the actual vid_length
                vid_dict[vid_clip_name]['frame_mask'] = [(frame_index <= vid_length) for frame_index in frame_index_list]
                frame_index_list = [min(frame_index, vid_length) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['backward_distance'] = [(frame_index - start_keyframe_index) / max(1, end_keyframe_index - start_keyframe_index) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['frame_path'] = ['frame{:06d}.png'.format(frame_index) for frame_index in frame_index_list]
                # normalize input_index by the original vid_length to [0, 1]
                vid_dict[vid_clip_name]['input_index'] = [(frame_index - 1) / (vid_length - 1) for frame_index in frame_index_list]
                self.frame_path_list.append([vid_clip_name, vid_name, vid_dict[vid_clip_name]['frame_path']])
                self.frame_count_total += clip_size

        self.vid_dict = vid_dict
        self.vid_list = sorted(list(vid_dict.keys()))
        self.frame_path_list = sorted(self.frame_path_list)

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        vid_clip_name = self.vid_list[idx]
        vid_name = self.vid_dict[vid_clip_name]['vid_name']

        frame_list = []
        for k in range(len(self.vid_dict[vid_clip_name]['frame_path'])):
            frame_path = self.vid_dict[vid_clip_name]['frame_path'][k]
            frame = Image.open(os.path.join(self.gt_base_dir, vid_name, frame_path)).convert("RGB")
            frame_list.append(self.transform_rgb(frame))
        video = torch.stack(frame_list, dim=1)

        input_index = torch.tensor(self.vid_dict[vid_clip_name]['input_index'])

        start_keyframe = self.transform_keyframe(Image.open(os.path.join(self.keyframe_base_dir, vid_name, self.vid_dict[vid_clip_name]['keyframe_path'][0])).convert("RGB"))
        end_keyframe = self.transform_keyframe(Image.open(os.path.join(self.keyframe_base_dir, vid_name, self.vid_dict[vid_clip_name]['keyframe_path'][1])).convert("RGB"))
        keyframe = torch.stack([start_keyframe, end_keyframe], dim=1)

        backward_distance = torch.tensor(self.vid_dict[vid_clip_name]['backward_distance'])
        frame_mask = torch.tensor(self.vid_dict[vid_clip_name]['frame_mask'])
        return video, input_index, keyframe, backward_distance, frame_mask

class Dataset_NeRV_UVG(Dataset):
    def __init__(self, args, transform_rgb, transform_keyframe=None):
        self.gt_base_dir = './data/UVG/gt'
        self.transform_rgb = transform_rgb

        vid_length_dict = collections.OrderedDict()
        with open('./data/UVG/annotation/video_length.json', 'r') as fp:
            vid_length_dict = json.load(fp)

        clip_size = args.clip_size

        vid_dict = collections.OrderedDict()
        self.frame_count_total = 0
        self.frame_path_list = []
        for vid_name, vid_length in vid_length_dict.items():
            # we divide videos into consecutive video_clips
            num_clip = round(math.ceil(vid_length / clip_size))
            # rounded up the vid_length, in case the vid_length is not divided by the clip_size
            vid_length_round = num_clip * clip_size

            for clip_index in range(num_clip):
                # the first frame is the start_keyframe, the first frame for the next consecutive clip is the end_keyframe
                start_keyframe_index = clip_index * clip_size + 1
                end_keyframe_index = min(vid_length, (clip_index + 1) * clip_size + 1)

                vid_clip_name = "{}-{}".format(vid_name, clip_index)
                vid_dict[vid_clip_name] = {}
                vid_dict[vid_clip_name]['vid_name'] = vid_name
                vid_dict[vid_clip_name]['keyframe_path'] = ['frame{:06d}.png'.format(start_keyframe_index), 'frame{:06d}.png'.format(end_keyframe_index)]
                frame_index_list = list(range(clip_index * clip_size + 1, (clip_index + 1) * clip_size + 1))
                # mask out the frame_index which are longer than the actual vid_length
                vid_dict[vid_clip_name]['frame_mask'] = [(frame_index <= vid_length) for frame_index in frame_index_list]
                frame_index_list = [min(frame_index, vid_length) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['backward_distance'] = [(frame_index - start_keyframe_index) / max(1, end_keyframe_index - start_keyframe_index) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['frame_path'] = ['frame{:06d}.png'.format(frame_index) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['input_index'] = [self.frame_count_total + i for i in range(clip_size)]
                self.frame_path_list.append([vid_clip_name, vid_name, vid_dict[vid_clip_name]['frame_path']])
                self.frame_count_total += clip_size

        self.vid_dict = vid_dict
        self.vid_list = sorted(list(vid_dict.keys()))
        self.frame_path_list = sorted(self.frame_path_list)

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        vid_clip_name = self.vid_list[idx]
        vid_name = self.vid_dict[vid_clip_name]['vid_name']

        frame_list = []
        for k in range(len(self.vid_dict[vid_clip_name]['frame_path'])):
            frame_path = self.vid_dict[vid_clip_name]['frame_path'][k]
            frame = Image.open(os.path.join(self.gt_base_dir, vid_name, frame_path)).convert("RGB")
            frame_list.append(self.transform_rgb(frame))
        video = torch.stack(frame_list, dim=1)

        input_index = torch.tensor(self.vid_dict[vid_clip_name]['input_index']) / (self.frame_count_total - 1)

        keyframe = torch.zeros(1)

        backward_distance = torch.zeros(1)
        frame_mask = torch.tensor(self.vid_dict[vid_clip_name]['frame_mask'])
        return video, input_index, keyframe, backward_distance, frame_mask

class Dataset_DNeRV_UCF101(Dataset):
    def __init__(self, args, transform_rgb, transform_keyframe=None):
        self.gt_base_dir = './data/UCF101/gt'
        self.keyframe_base_dir = './data/UCF101/keyframe/q{}'.format(args.keyframe_quality)
        self.transform_rgb = transform_rgb
        self.transform_keyframe = transform_keyframe

        vid_length_dict = collections.OrderedDict()
        with open('./data/UCF101/annotation/video_length_train.json', 'r') as fp:
            vid_length_dict = json.load(fp)

        clip_size = args.clip_size

        vid_dict = collections.OrderedDict()
        self.frame_count_total = 0
        self.frame_path_list = []
        for vid_name, vid_length in vid_length_dict.items():
            action_name = vid_name.split('_')[1]
            num_clip = round(math.ceil(vid_length / clip_size))
            vid_length_round = num_clip * clip_size
            for clip_index in range(num_clip):
                start_keyframe_index = clip_index * clip_size + 1
                end_keyframe_index = min(vid_length, (clip_index + 1) * clip_size + 1)
                vid_clip_name = "{}-{}".format(vid_name, clip_index)
                vid_dict[vid_clip_name] = {}
                vid_dict[vid_clip_name]['vid_name'] = vid_name
                vid_dict[vid_clip_name]['action_name'] = action_name
                vid_dict[vid_clip_name]['keyframe_path'] = ['frame{:06d}.png'.format(start_keyframe_index), 'frame{:06d}.png'.format(end_keyframe_index)]
                frame_index_list = list(range(clip_index * clip_size + 1, (clip_index + 1) * clip_size + 1))
                vid_dict[vid_clip_name]['frame_mask'] = [(frame_index <= vid_length) for frame_index in frame_index_list]
                frame_index_list = [min(frame_index, vid_length) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['backward_distance'] = [(frame_index - start_keyframe_index) / max(1, end_keyframe_index - start_keyframe_index) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['frame_path'] = ['frame{:06d}.png'.format(frame_index) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['input_index'] = [(frame_index - 1) / (vid_length - 1) for frame_index in frame_index_list]
                self.frame_path_list.append([vid_clip_name, action_name, vid_name, vid_dict[vid_clip_name]['frame_path']])
                self.frame_count_total += clip_size

        self.vid_dict = vid_dict
        self.vid_list = sorted(list(vid_dict.keys()))
        self.frame_path_list = sorted(self.frame_path_list)

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        vid_clip_name = self.vid_list[idx]
        vid_name = self.vid_dict[vid_clip_name]['vid_name']
        action_name = self.vid_dict[vid_clip_name]['action_name']

        frame_list = []
        for k in range(len(self.vid_dict[vid_clip_name]['frame_path'])):
            frame_path = self.vid_dict[vid_clip_name]['frame_path'][k]
            frame = Image.open(os.path.join(self.gt_base_dir, action_name, vid_name, frame_path)).convert("RGB")
            frame_list.append(self.transform_rgb(frame))
        video = torch.stack(frame_list, dim=1)

        input_index = torch.tensor(self.vid_dict[vid_clip_name]['input_index'])

        start_keyframe = self.transform_keyframe(Image.open(os.path.join(self.keyframe_base_dir, action_name, vid_name, self.vid_dict[vid_clip_name]['keyframe_path'][0])).convert("RGB"))
        end_keyframe = self.transform_keyframe(Image.open(os.path.join(self.keyframe_base_dir, action_name, vid_name, self.vid_dict[vid_clip_name]['keyframe_path'][1])).convert("RGB"))
        keyframe = torch.stack([start_keyframe, end_keyframe], dim=1)

        backward_distance = torch.tensor(self.vid_dict[vid_clip_name]['backward_distance'])
        frame_mask = torch.tensor(self.vid_dict[vid_clip_name]['frame_mask'])
        return video, input_index, keyframe, backward_distance, frame_mask

class Dataset_NeRV_UCF101(Dataset):
    def __init__(self, args, transform_rgb, transform_keyframe=None):
        self.gt_base_dir = './data/UCF101/gt'
        self.transform_rgb = transform_rgb

        vid_length_dict = collections.OrderedDict()
        with open('./data/UCF101/annotation/video_length_train.json', 'r') as fp:
            vid_length_dict = json.load(fp)

        clip_size = args.clip_size

        vid_dict = collections.OrderedDict()
        self.frame_count_total = 0
        self.frame_path_list = []
        for vid_name, vid_length in vid_length_dict.items():
            action_name = vid_name.split('_')[1]
            num_clip = round(math.ceil(vid_length / clip_size))
            vid_length_round = num_clip * clip_size
            for clip_index in range(num_clip):
                start_keyframe_index = clip_index * clip_size + 1
                end_keyframe_index = min(vid_length, (clip_index + 1) * clip_size + 1)
                vid_clip_name = "{}-{}".format(vid_name, clip_index)
                vid_dict[vid_clip_name] = {}
                vid_dict[vid_clip_name]['vid_name'] = vid_name
                vid_dict[vid_clip_name]['action_name'] = action_name

                frame_index_list = list(range(clip_index * clip_size + 1, (clip_index + 1) * clip_size + 1))
                vid_dict[vid_clip_name]['frame_mask'] = [(frame_index <= vid_length) for frame_index in frame_index_list]
                frame_index_list = [min(frame_index, vid_length) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['frame_path'] = ['frame{:06d}.png'.format(frame_index) for frame_index in frame_index_list]
                vid_dict[vid_clip_name]['input_index'] = [self.frame_count_total + i for i in range(clip_size)]
                self.frame_path_list.append([vid_clip_name, action_name, vid_name, vid_dict[vid_clip_name]['frame_path']])
                self.frame_count_total += clip_size

        self.vid_dict = vid_dict
        self.vid_list = sorted(list(vid_dict.keys()))
        self.frame_path_list = sorted(self.frame_path_list)

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        vid_clip_name = self.vid_list[idx]
        vid_name = self.vid_dict[vid_clip_name]['vid_name']
        action_name = self.vid_dict[vid_clip_name]['action_name']

        frame_list = []
        for k in range(len(self.vid_dict[vid_clip_name]['frame_path'])):
            frame_path = self.vid_dict[vid_clip_name]['frame_path'][k]
            frame = Image.open(os.path.join(self.gt_base_dir, action_name, vid_name, frame_path)).convert("RGB")
            frame_list.append(self.transform_rgb(frame))
        video = torch.stack(frame_list, dim=1)

        input_index = torch.tensor(self.vid_dict[vid_clip_name]['input_index']) / (self.frame_count_total - 1)

        keyframe = torch.zeros(1)

        backward_distance = torch.zeros(1)
        frame_mask = torch.tensor(self.vid_dict[vid_clip_name]['frame_mask'])
        return video, input_index, keyframe, backward_distance, frame_mask


def my_collate_fn(batch):
    batched_output_list = []
    for i in range(len(batch[0])):
        if torch.is_tensor(batch[0][i]):
            batched_output = torch.stack([single_batch[i] for single_batch in batch], dim=0)
        elif type(batch[0][i]) is dict:
            batched_output = {}
            for k, v in batch[0][i].items():
                batched_output[k] = torch.stack([single_batch[i][k] for single_batch in batch], dim=0)
        batched_output_list.append(batched_output)
    return batched_output_list

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return
