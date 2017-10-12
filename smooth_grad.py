#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-11

from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm


class SmoothGrad(object):

    def __init__(self, model, cuda, sigma=0.20,
                 n_samples=50, guidedbackprop=True):
        self.model = model
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.sigma = sigma
        self.n_samples = n_samples
        self.probs = None

        if guidedbackprop:
            def func_b(module, grad_in, grad_out):
                if isinstance(module, nn.ReLU):
                    return (F.threshold(grad_in[0], threshold=0.0, value=0.0),)

            for module in self.model.named_modules():
                module[1].register_backward_hook(func_b)

    def load_image(self, filename, transform):
        raw_image = cv2.imread(filename)[:, :, ::-1]
        raw_image = cv2.resize(raw_image, (224, 224))
        image = transform(raw_image).unsqueeze(0)
        if self.cuda:
            image = image.cuda()
        self.image = Variable(image, volatile=False, requires_grad=True)

    def forward(self):
        self.preds = self.model.forward(self.image)
        self.probs = F.softmax(self.preds)[0]
        self.prob, self.idx = self.probs.data.sort(0, True)

    def encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.probs.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self.encode_one_hot(idx)
        if self.cuda:
            one_hot = one_hot.cuda()
        self.preds.backward(gradient=one_hot, retain_graph=True)

    def generate(self, idx, filename):
        grads = []
        image = self.image.data.cpu()
        sigma = (image.max() - image.min()) * self.sigma

        for i in tqdm(range(self.n_samples)):
            # Add gaussian noises
            noised_image = image + torch.randn(image.size()) * sigma
            self.image = Variable(
                noised_image, volatile=False, requires_grad=True)
            self.forward()
            self.backward(idx=idx)
            grad = self.image.grad.data.cpu().numpy()
            grads.append(grad)
            self.model.zero_grad()

            if i % 5 == 0:
                grad = np.mean(np.array(grads), axis=0)
                img = np.max(np.abs(grad), axis=1)[0]
                img -= img.min()
                img /= img.max()
                img = np.uint8(img * 255)
                cv2.imwrite(filename + '_{}.png'.format(i), img)
