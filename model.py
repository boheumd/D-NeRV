import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import pi, sqrt
from utils import WarpKeyframe


class MLP(nn.Module):
    def __init__(self, in_chan, out_chan, hidden_chan=512, act='GELU', bias=True, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_chan, hidden_chan, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv1d(hidden_chan, out_chan, 1, 1, 0, bias=bias),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, kernel_size=3, stride=1, stride_list=[], bias=True):
        super().__init__()
        n_resblocks = len(stride_list)

        # define head module
        m_head = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, kernel_size//2, kernel_size//2), bias=bias),
            nn.GELU(),
        )
        m_body = []
        for i in range(n_resblocks):
            m_body.append(nn.Sequential(
                            nn.Conv3d(64, 64, kernel_size=(1, stride_list[i], stride_list[i]), stride=(1, stride_list[i], stride_list[i]), padding=(0, 0, 0), bias=bias),
                            nn.GELU(),
                            )
                        )
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)

    def forward(self, x):
        key_feature_list = [x]
        x = self.head(x)
        for stage in self.body:
           x = stage(x)
           key_feature_list.append(x)
        return key_feature_list[::-1]

class Head(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(in_chan, in_chan // 4, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
                        nn.GELU(),
                        nn.Conv3d(in_chan // 4, out_chan, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
                    )

    def forward(self, x):
        x = self.conv(x)
        return x

class ToStyle(nn.Module):
    def __init__(self, in_chan, out_chan, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_chan, out_chan*2, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)

    def forward(self, x, style):
        B, C, T, H, W = x.shape
        style = self.conv(style)  # style -> [B, 2*C, 1, H, W]
        style = style.view(B, 2, C, -1, H, W)  # [B, 2, C, 1, H, W]
        x = x * (style[:, 0] + 1.) + style[:, 1] # [B, C, T, H, W]
        return x

class PixelShuffle(nn.Module):
    def __init__(self, scale=(1, 2, 2)):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        B, C, T, H, W = x.size()
        C_out = C // (self.scale[0] * self.scale[1] * self.scale[2])
        x = x.view(B, C_out, self.scale[0], self.scale[1], self.scale[2], T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(B, C_out, T * self.scale[0], H * self.scale[1], W * self.scale[2])
        return x


class DNeRVBlock(nn.Module):
    def __init__(self, kernel=3, bias=True, **kwargs):
        super().__init__()
        in_chan = kwargs['ngf']
        out_chan = kwargs['new_ngf'] * kwargs['stride'] * kwargs['stride']
        # Spatially-adaptive Fusion
        self.to_style = ToStyle(64, in_chan, bias=bias)
        # 3x3 Convolution-> PixelShuffle -> Activation, same as NeRVBlock
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size=(1, kernel, kernel), stride=(1, 1, 1), padding=(0, kernel//2, kernel//2), bias=bias)
        self.upsample = PixelShuffle(scale=(1, kwargs['stride'], kwargs['stride']))
        self.act = nn.GELU()
        # Global Temporal MLP module
        self.tfc = nn.Conv2d(kwargs['new_ngf']*kwargs['clip_size'], kwargs['new_ngf']*kwargs['clip_size'], 1, 1, 0, bias=True, groups=kwargs['new_ngf'])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv3d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, x, style_appearance):
        x = self.to_style(x, style_appearance)
        x = self.act(self.upsample(self.conv(x)))
        B, C, D, H, W = x.shape
        x = x + self.tfc(x.view(B, C*D, H, W)).view(B, C, D, H, W)
        return x

class DNeRV(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        ngf = kwargs['fc_dim']
        self.stem = MLP(in_chan=kwargs['embed_length'], out_chan=self.fc_h * self.fc_w * ngf)

        self.stride_list = kwargs['stride_list']
        self.num_stages = len(self.stride_list)

        encoder_dim = 64
        self.encoder = Encoder(stride_list=self.stride_list[::-1])
        self.norm = nn.InstanceNorm3d(ngf + encoder_dim)

        self.decoder_list = nn.ModuleList()
        self.flow_pred_list = nn.ModuleList([Head(ngf + encoder_dim, 4)])

        height = self.fc_h
        width = self.fc_w
        self.wk_list = nn.ModuleList([WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device'])])
        for i, stride in enumerate(self.stride_list):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                new_ngf = max(round(ngf / stride), kwargs['lower_width'])

            self.decoder_list.append(DNeRVBlock(ngf=ngf + encoder_dim if i == 0 else ngf, new_ngf=new_ngf, 
                                                stride=stride, clip_size=kwargs['clip_size']))
            self.flow_pred_list.append(Head(new_ngf, 4))
            height = height * stride
            width = width * stride
            self.wk_list.append(WarpKeyframe(height, width, kwargs['clip_size'], device=kwargs['device']))

            ngf = new_ngf
        
        self.rgb_head_layer = Head(new_ngf + 3, 3)

        self.dataset_mean = torch.tensor(kwargs['dataset_mean']).view(1, 3, 1, 1, 1).to(kwargs['device'])
        self.dataset_std = torch.tensor(kwargs['dataset_std']).view(1, 3, 1, 1, 1).to(kwargs['device'])

    def forward(self, embed_input, keyframe, backward_distance):
        B, C, D = embed_input.size()
        backward_distance = backward_distance.view(B, 1, -1, 1, 1)
        forward_distance = 1 - backward_distance

        key_feature_list = self.encoder(keyframe) # [B, encoder_dim, 2, H, W]
        output = self.stem(embed_input)  # [B, C*fc_h*fc_w, D]
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 1, 4, 2, 3)  # [B, C, D, fc_h, fc_w]
        content_feature = F.interpolate(key_feature_list[0], scale_factor=(D/2, 1, 1), mode='trilinear') # [B, encoder_dim, D, fc_h, fc_w]
        output = self.norm(torch.concat([output, content_feature], dim=1))

        for i in range(self.num_stages + 1):
            # generate flow at the decoder input stage
            flow = self.flow_pred_list[i](output) # [B, 4, D, fc_h, fc_w]
            forward_flow, backward_flow = torch.split(flow, [2, 2], dim=1)
            start_key_feature, end_key_feature = torch.split(key_feature_list[i], [1, 1], dim=2)
            # warp the keyframe features with predicted forward and backward flow
            forward_warp = self.wk_list[i](start_key_feature, forward_flow)
            backward_warp = self.wk_list[i](end_key_feature, backward_flow)
            # distance-aware weighted sum
            fused_warp = forward_warp * forward_distance + backward_warp * backward_distance # (1 - t) * forward_warp + t * backward_warp

            if i < self.num_stages:
                output = self.decoder_list[i](output, fused_warp)
            else:
                output = self.rgb_head_layer(torch.cat([output, fused_warp], dim=1))

        output = output * self.dataset_std + self.dataset_mean
        return output.clamp(min=0, max=1)


class NeRVBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(kwargs['ngf'], kwargs['new_ngf']*kwargs['stride']*kwargs['stride'], 3, 1, 1)
        self.up_scale = nn.PixelShuffle(kwargs['stride'])
        self.act = nn.GELU()

    def forward(self, x, sty=None):
        x = self.act(self.up_scale(self.conv(x)))
        return x

class NeRV(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc_h, self.fc_w = [int(x) for x in kwargs['fc_hw'].split('_')]
        ngf = kwargs['fc_dim']
        self.stem = MLP(in_chan=kwargs['embed_length'], out_chan=self.fc_h * self.fc_w * ngf)
        
        self.layers = nn.ModuleList()
        for i, stride in enumerate(kwargs['stride_list']):
            if i == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                reduction = stride
                new_ngf = max(round(ngf / reduction), kwargs['lower_width'])

            self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride))
            ngf = new_ngf

        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1) 

    def forward(self, embed_input):
        B, C, D = embed_input.size()

        output = self.stem(embed_input) # [B, C*fc_h*fc_w, D]
        output = output.view(B, -1, self.fc_h, self.fc_w, D).permute(0, 4, 1, 2, 3)  # [B, D, C, fc_h, fc_w]
        output = output.reshape(B*D, -1, self.fc_h, self.fc_w)
        out_list = []
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        output = self.head_layer(output)
        output = (torch.tanh(output) + 1) * 0.5

        BD, C, H, W = output.size()
        output = output.view(B, D, C, H, W).permute(0, 2, 1, 3, 4) # [B, C, D, H, W]
        return  output
