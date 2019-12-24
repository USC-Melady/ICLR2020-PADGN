from __future__ import division
from __future__ import print_function

import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)

import socket
import time
import argparse
import pickle
import datetime
import multiprocessing

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as torchF
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter


def sample_todevice(sample, device):
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v.contiguous().to(device, non_blocking=True)
        elif isinstance(v, list):
            templist = []
            for vi in v:
                if isinstance(vi, torch.Tensor):
                    templist.append(vi.contiguous().to(device, non_blocking=True))
                else:
                    templist.append(vi)
            sample[k] = templist
    return sample


class MyArgs:
    def __init__(self, **argdict):
        for k, v in argdict.items():
            if isinstance(v, dict):
                self.__dict__[k] = MyArgs(**v)
            else:
                self.__dict__[k] = v

    def to_argdict(self):
        argdict = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, MyArgs):
                argdict[k] = v.to_argdict()
            else:
                argdict[k] = v
        return argdict

    def load_argdict(self, argdict):
        for k, v in argdict.items():
            if isinstance(v, dict):
                self.__dict__[k] = MyArgs(**v)
            else:
                self.__dict__[k] = v


def get_last_ckpt(ckptdir, device, suffix='_checkpoint.pt', specify=None):
    if specify is not None:
        # last_ckpt = torch.load(os.path.join(ckptdir, '{:d}'.format(specify) + suffix))
        last_ckpt = torch.load(os.path.join(ckptdir, '{}'.format(specify) + suffix))
    else:
        ckpts = []
        for x in os.listdir(ckptdir):
            if x.endswith(suffix) and (not x.startswith('best_')):
                xs = x.replace(suffix, '')
                ckpts.append((x, int(xs)))
        if len(ckpts) == 0:
            last_ckpt = None
        else:
            ckpts.sort(key=lambda x: x[1])
            last_ckpt = torch.load(os.path.join(ckptdir, ckpts[-1][0]), map_location=device)
    if os.path.exists(os.path.join(ckptdir, 'best' + suffix)):
        best_ckpt = torch.load(os.path.join(ckptdir, 'best' + suffix), map_location=device)
    else:
        best_ckpt = None
    return {
        'last': last_ckpt, 'best': best_ckpt
    }


def save_ckpt(epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler, ckptdir,
              prefix, suffix='_checkpoint.pt'):
    ckptdict = {
        'epoch': epoch,
        'best_valid_loss': best_valid_loss,
        'best_valid_epoch': best_valid_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(ckptdict, os.path.join(ckptdir, prefix + suffix))
    return ckptdict


def load_ckpt(model, optimizer, scheduler, ckpt, restore_opt_sche=True):
    epoch = ckpt['epoch']
    best_valid_loss = ckpt['best_valid_loss']
    best_valid_epoch = ckpt['best_valid_epoch']
    model.load_state_dict(ckpt['model'])
    if restore_opt_sche:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    return epoch, best_valid_loss, best_valid_epoch, model, optimizer, scheduler


def print_2way(f, *x):
    print(*x)
    print(*x, file=f)
    f.flush()


class SampleScheduler(object):
    def __init__(self, name):
        self._name = name

    def get_train_sample_prob(self, epoch):
        raise NotImplementedError()


class AlwaysSampleScheduler(SampleScheduler):
    def __init__(self):
        super(AlwaysSampleScheduler, self).__init__('AlwaysSampleScheduler')

    def get_train_sample_prob(self, epoch):
        return 1.0


class InverseSigmoidDecaySampleScheduler(SampleScheduler):
    def __init__(self, epochnum, delay_start=0):
        super(InverseSigmoidDecaySampleScheduler, self).__init__('InverseSigmoidDecaySampleScheduler')
        self._delay_start = delay_start  # start scheduling after delay_start epochs
        self._k = self.solve_k((epochnum - self._delay_start) / 2) # train_sample_prob ~ 0.5 when epoch_i = epochnum / 2

    def get_train_sample_prob(self, epoch):
        if epoch < self._delay_start:
            return 1
        else:
            epoch = epoch - self._delay_start
            return self._k / (self._k + np.exp(epoch / self._k))

    def solve_k(self, a):
        '''
        Using Newton's method to solve klnk = a, used for InverseSigmoidDecay
        :param epochnum:
        :return: s, such that s * ln(s) - a = 0
        '''
        s = 2 * a
        while True:
            news = (s + a) / (np.log(s) + 1)
            if np.abs(news - s) < 1e-6:
                return s
            else:
                s = news