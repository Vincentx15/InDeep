#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import partial

if __name__ == '__main__':
    import sys

    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from learning.BaseModel import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d_a = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                  stride=stride, padding=padding)
        self.conv3d_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                                  stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d_a(x))
        x = self.conv3d_b(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2, num_feat_maps=16):
        super(EncoderBlock, self).__init__()
        self.num_feat_maps = num_feat_maps
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        downsampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                if k.endswith("1"):
                    downsampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)

        return x, downsampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    """
    Constructs back the input like grids from the condensed representation and the intermediate values.
    The final size of the map is
    """

    def __init__(self, out_channels, model_depth=4, num_feat_maps=16, double_conv=False):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = num_feat_maps
        self.double_conv = double_conv
        # user nn.ModuleDict() to store ops and the fact that the order is kept
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                if self.double_conv:
                    self.final_conv_hd = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                    self.final_conv_pl = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                    self.final_convs = [self.final_conv_hd, self.final_conv_pl]
                else:
                    self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                    self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, downsampling_features, branch=0):
        """
        :param x: inputs
        :param downsampling_features: feature maps from encoder path
        :return: output
        """

        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                # If the input has a shape that is not a power of 2, we need to pad when deconvoluting
                *_, dx, dy, dz = [a - b for a, b in zip((downsampling_features[int(k[-1])].shape), (x.shape))]
                pad_size = [0, dz, 0, dy, 0, dx]
                x = torch.nn.functional.pad(x, pad=pad_size, mode='constant', value=0)
                x = torch.cat((downsampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        if self.double_conv:
            x = self.final_convs[branch](x)
        return x


def unet_from_hparams(hparams, load_weights=False):
    out_channels = hparams.get('argparse', 'out_channels')
    model_depth = hparams.get('argparse', 'model_depth')
    double_conv = hparams.get('argparse', 'double_conv')

    try:
        num_feature_map = hparams.get('argparse', 'num_feature_map')
    except KeyError:
        print("Failed loading num feature maps")
        num_feature_map = 16

    batch_size = hparams.get('argparse', 'batch_size')
    wpl = hparams.get('pl', 'wpl')
    whd = hparams.get('hd', 'whd')
    wenveloppe = hparams.get('common', 'wenveloppe')
    unet = UnetModel(out_channels=out_channels,
                     model_depth=model_depth,
                     wpl=wpl,
                     whd=whd,
                     wenveloppe=wenveloppe,
                     batch_size=batch_size,
                     double_conv=double_conv,
                     num_feature_map=num_feature_map)
    if load_weights:
        unet.load_weights(hparams=hparams)
    return unet


class UnetModel(BaseModel):

    def __init__(self,
                 out_channels=32,
                 wpl=0.05,
                 whd=0.05,
                 wenveloppe=0.35,
                 in_channels=5,
                 model_depth=4,
                 num_feature_map=16,
                 batch_size=2,
                 double_conv=False):
        super(UnetModel, self).__init__()

        self.wpl = wpl
        self.whd = whd
        self.wenveloppe = wenveloppe
        self.score_ones = [whd, wpl]
        self.flush_size = batch_size
        self.double_conv = double_conv
        self.num_feat_maps = num_feature_map
        self.encoder = EncoderBlock(in_channels=in_channels,
                                    model_depth=model_depth,
                                    num_feat_maps=self.num_feat_maps)
        self.decoder = DecoderBlock(out_channels=out_channels,
                                    model_depth=model_depth,
                                    double_conv=double_conv,
                                    num_feat_maps=self.num_feat_maps)

        self.optimizer = optim.Adam(self.parameters())
        self.passes = 0

        mid_channels = max(out_channels // 2, 5)
        self.conv3d_hd = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_channels, out_channels=6, kernel_size=1),
        )

        self.conv3d_pl = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_channels, out_channels=1, kernel_size=1),
        )
        self.branch_final = [self.conv3d_hd, self.conv3d_pl]

        act_hd = partial(nn.functional.softmax, dim=1)

        def act_pl(inputs):
            return torch.clamp(torch.sigmoid(inputs), 1e-5, 1 - 1e-5)

        self.branch_acts = [act_hd, act_pl]
        self.branch_losses = [self.weighted_CE_loss, self.weighted_BCE_loss]
        # self.branch_losses = [self.weighted_dice_loss, self.weighted_binary_dice_loss]

    def decode_branch(self, mid, downsampling_features, branch):
        x = self.decoder(mid, downsampling_features=downsampling_features, branch=branch)
        x = self.branch_final[branch](x)
        x = self.branch_acts[branch](x)
        return x

    # For jit compilation and porting to torchscript
    # def forward(self, x):
    #     mid, downsampling_features = self.encoder(x)
    #     out = self.decode_branch(mid=mid, downsampling_features=downsampling_features, branch=1)
    #     return out

    def forward(self, x, branch):
        mid, downsampling_features = self.encoder(x)
        out = self.decode_branch(mid=mid, downsampling_features=downsampling_features, branch=branch)
        return out

    def inference_call(self, inputs):
        """
        Just infer all branch labelling for a given input
        :param inputs:
        :return:
        """
        self.eval()

        with torch.no_grad():
            mid, downsampling_features = self.encoder(inputs)
            hd = self.decode_branch(mid=mid, downsampling_features=downsampling_features, branch=0)
            pl = self.decode_branch(mid=mid, downsampling_features=downsampling_features, branch=1)
        return hd, pl

    def testing_call(self, inputs, y, branch):
        """
        Just infer a branch labelling for a given input
        :param inputs:
        :return:
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(inputs, branch=branch)

            # Loss computation
            loss_obj = self.branch_losses[branch]
            loss = loss_obj(out, y)
        return out, loss.item()

    def persistent_training_call(self, inputs, y, branch, enveloppe=None, frozen_common=False):
        """
        Perform the right forward pass, the right backward and returns the loss value as well as the output
        :param inputs:
        :param branch:
        :return:
        """
        # print('using old unet')
        self.train()

        # We might want to freeze the hd results to focus on PL.
        out = self.forward(inputs, branch=branch)

        # Loss computation TODO : careful with batches !
        if branch == 0:
            target_scale = 1 - y[0, -1, ...]
        else:
            target_scale = y[0, 0, ...]
        weight_loss = self.compute_weight(target=target_scale,
                                          enveloppe=enveloppe,
                                          scale_ones=self.score_ones[branch],
                                          scale_enveloppe=self.wenveloppe)

        loss_func = self.branch_losses[branch]
        loss = loss_func(output=out, target=y, weight=weight_loss)
        loss = loss / self.flush_size
        loss.backward()

        self.passes += 1
        if self.passes > self.flush_size:
            self.passes = 0
            self.optimizer.step()
            self.optimizer.zero_grad()

        return out, loss.item()


if __name__ == '__main__':
    model = UnetModel(double_conv=True)
    print(model)
    grid_prot, grid_hd, grid_pl = torch.ones((1, 5, 5, 5, 5), dtype=torch.float32), \
                                  torch.ones((1, 6, 5, 5, 5), dtype=torch.float32), \
                                  torch.ones((1, 1, 5, 5, 5), dtype=torch.float32)
    # out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    # out, loss = model.persistent_training_call(grid_prot, grid_pl, 1)
