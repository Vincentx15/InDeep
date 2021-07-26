#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2019-09-12 09:42:09 (UTC+0200)

import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from learning.BaseModel import BaseModel


def c3d_from_hparams(hparams, load_weights=False):
    common_in_size = hparams.get('common', 'in_size')
    common_filters = hparams.get('common', 'filters')
    common_drate = hparams.get('common', 'drate')

    hdnet = hdnet_from_hparams(hparams)
    plnet = plnet_from_hparams(hparams)

    flush_size = hparams.get('argparse', 'batch_size')

    try:
        wpl = hparams.get('pl', 'wpl')
        whd = hparams.get('hd', 'whd')
        wenveloppe = hparams.get('common', 'wenveloppe')
    except KeyError:
        print("Before the weighting of the loss, network was loaded but can't be fine-tuned")
        wpl = 0
        whd = 0
        wenveloppe = 0

    c3d = C3D(common_in_size=common_in_size,
              common_filters=common_filters,
              common_drate=common_drate,
              hdnet=hdnet,
              plnet=plnet,
              wpl=wpl,
              whd=whd,
              wenveloppe=wenveloppe,
              flush_size=flush_size)
    if load_weights:
        c3d.load_weights(hparams=hparams)
    return c3d


# C3D Model
class C3D(BaseModel):

    def __init__(self,
                 common_in_size,
                 common_filters,
                 common_drate,
                 hdnet,
                 plnet,
                 wpl,
                 whd,
                 wenveloppe,
                 flush_size):
        super(C3D, self).__init__()

        self.common = self.build_common(in_size=common_in_size,
                                        filters=common_filters,
                                        drate=common_drate)
        self.hd = hdnet
        self.pl = plnet
        self.branches = [self.hd, self.pl, self.common]

        # self.optimizers = [optim.SGD(self.hd.parameters(), lr=0.01),
        #                    optim.SGD(self.pl.parameters(), lr=0.01),
        #                    optim.SGD(self.common.parameters(), lr=0.01)]

        self.optimizers = [optim.Adam(self.hd.parameters()),
                           optim.Adam(self.pl.parameters()),
                           optim.Adam(self.common.parameters())]
        self.branch_losses = [self.weighted_CE_loss, self.weighted_BCE_loss]
        self.wpl = wpl
        self.whd = whd
        self.score_ones = [whd, wpl]
        self.wenveloppe = wenveloppe

        self.passes = [0, 0, 0]
        self.flush_size = flush_size

    @staticmethod
    def build_common(in_size, filters, drate):
        """
        Build the common CNN
        """
        layers = list()
        for layer_id, filters_ in enumerate(filters):
            if layer_id == 0:  # Input layer
                layers.append(nn.Conv3d(in_size, filters_,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.Dropout(drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))

            else:
                layers.append(nn.Conv3d(previous, filters_,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.Dropout(drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            previous = filters_
        return nn.Sequential(*layers)

    def eval(self):
        self.common.eval()
        self.hd.eval()
        self.pl.eval()

    def train(self, mode=True):
        self.common.train(mode)
        self.hd.train(mode)
        self.pl.train(mode)

    def compute_grads(self, hd=True, pl=True, common=True):
        self.common.requires_grad = common
        self.hd.requires_grad = hd
        self.pl.requires_grad = pl

    def inference_call(self, inputs):
        """
        Just infer all branch labelling for a given input
        :param inputs:
        :return:
        """
        self.eval()
        with torch.no_grad():
            mid = self.common(inputs)
            hd = self.hd(mid)
            pl = self.pl(mid)
        return hd, pl

    def testing_call(self, inputs, y, branch):
        """
        Just infer a branch labelling for a given input
        :param inputs:
        :return:
        """
        self.eval()
        with torch.no_grad():
            mid = self.common(inputs)
            out = self.branches[branch](mid)

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

        self.train()

        # We might want to freeze the hd results to focus on PL.
        if frozen_common:
            with torch.no_grad():
                mid = self.common(inputs)
        else:
            mid = self.common(inputs)

        out = self.branches[branch].forward(mid)

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

        # Only try to backprop if it's not frozen
        if not frozen_common:
            self.passes[-1] += 1
        self.passes[branch] += 1

        if self.passes[-1] > self.flush_size:
            self.passes[-1] = 0
        self.optimizers[-1].step()
        self.optimizers[-1].zero_grad()

        if self.passes[branch] > self.flush_size:
            self.passes[branch] = 0
        self.optimizers[branch].step()
        self.optimizers[branch].zero_grad()

        return out, loss.item()


def hdnet_from_hparams(hparams):
    use_connected = hparams.get('hd', 'connected')
    mid_size = hparams.get('common', 'filters')[-1]
    drate = hparams.get('hd', 'drate')
    filters = hparams.get('hd', 'filters')
    final_convolution = hparams.get('hd', 'filters')[-1]
    connected_drate = hparams.get('connected', 'drate')
    connected_filters = hparams.get('connected', 'filters')
    hdnet = HDNet(use_connected=use_connected,
                  mid_size=mid_size,
                  drate=drate,
                  filters=filters,
                  final_convolution=final_convolution,
                  connected_filters=connected_filters,
                  connected_drate=connected_drate)
    return hdnet


class HDNet(nn.Module):
    """
    We use this wrapper class to deal with the activation functions that cannot be included in a
    Sequential like in TF2... So we have to write the forward by hand
    """

    def __init__(self,
                 use_connected,
                 mid_size,
                 drate,
                 filters,
                 final_convolution,
                 connected_filters,
                 connected_drate):
        super(HDNet, self).__init__()

        self.use_connected = use_connected
        self.net = self.build_conv(mid_size=mid_size, drate=drate, filters=filters)
        if self.use_connected:
            self.connected = self.build_connected(final_convolution=final_convolution,
                                                  connected_drate=connected_drate,
                                                  connected_filters=connected_filters)
        else:
            self.connected = None

    @staticmethod
    def build_conv(mid_size, drate, filters):
        """
        Build the HD CNN
        """

        layers = list()
        for layer_id, filters_ in enumerate(filters):
            if layer_id == 0:  # Input layer
                layers.append(nn.Conv3d(mid_size, filters_,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.Dropout(drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            elif layer_id < len(filters) - 1:
                layers.append(nn.Conv3d(previous, filters_,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.Dropout(drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            else:
                layers.append(nn.Conv3d(previous, filters_,
                                        kernel_size=3,
                                        padding=1))
            previous = filters_
        return nn.Sequential(*layers)

    @staticmethod
    def build_connected(final_convolution, connected_drate, connected_filters):
        """
        Build the connected one
        """
        layers = list()
        for layer_id, filters_ in enumerate(connected_filters):
            if layer_id == 0:  # Input layer
                layers.append(nn.Conv3d(final_convolution, filters_,
                                        kernel_size=1))
                layers.append(nn.Dropout(connected_drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            elif layer_id < len(connected_filters) - 1:
                layers.append(nn.Conv3d(previous, filters_,
                                        kernel_size=1))
                layers.append(nn.Dropout(connected_drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            else:
                layers.append(nn.Conv3d(previous, filters_,
                                        kernel_size=1))
            previous = filters_
        return nn.Sequential(*layers)

    def forward(self, a):
        out = self.net(a)
        if self.use_connected:
            out = self.connected(out)

        # Simple version
        out = nn.functional.softmax(out, dim=1)
        return out

        # distrib = out[..., :-1, :, :, :]
        # proba = out[..., -1, :, :, :]
        # proba = proba[None, ...]
        # distrib = nn.functional.softmax(distrib, dim=1)
        # proba = nn.functional.sigmoid(proba)
        # out = torch.cat((distrib, proba), dim=1)
        # return out


def plnet_from_hparams(hparams):
    mid_size = hparams.get('common', 'filters')[-1]
    drate = hparams.get('pl', 'drate')
    filters = hparams.get('pl', 'filters')
    plnet = PLNet(mid_size=mid_size,
                  drate=drate,
                  filters=filters)
    return plnet


class PLNet(nn.Module):
    def __init__(self,
                 mid_size,
                 drate,
                 filters):
        super(PLNet, self).__init__()

        self.eps = 1e-5

        self.net = self.build_pl(mid_size=mid_size,
                                 drate=drate,
                                 filters=filters)

    @staticmethod
    def build_pl(mid_size,
                 drate,
                 filters):
        """
        Build the common CNN
        """

        layers = list()
        for layer_id, filters_ in enumerate(filters):
            if layer_id == 0:  # Input layer
                layers.append(nn.Conv3d(mid_size, filters_,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.Dropout(drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            elif layer_id < len(filters) - 1:
                layers.append(nn.Conv3d(previous, filters_,
                                        kernel_size=3,
                                        padding=1))
                layers.append(nn.Dropout(drate))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            else:
                layers.append(nn.Conv3d(previous, filters_,
                                        kernel_size=3,
                                        padding=1))
            previous = filters_
        return nn.Sequential(*layers)

    def forward(self, a):
        out = self.net(a)
        out = torch.sigmoid(out)
        out = torch.clamp(out, self.eps, 1 - self.eps)
        return out


if __name__ == '__main__':
    model = C3D()
    grid_prot, grid_hd, grid_pl = torch.ones((1, 8, 5, 5, 5), dtype=torch.float32), \
                                  torch.ones((1, 9, 5, 5, 5), dtype=torch.float32), \
                                  torch.ones((1, 1, 5, 5, 5), dtype=torch.float32)
    out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    out, loss = model.persistent_training_call(grid_prot, grid_pl, 1)
    out, loss = model.persistent_training_call(grid_prot, grid_pl, 1)
    out, loss = model.persistent_training_call(grid_prot, grid_pl, 1)
