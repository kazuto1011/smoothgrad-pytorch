#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-11

from __future__ import print_function

import argparse

import torch

import torchvision
from smooth_grad import SmoothGrad
from torchvision import transforms


def main(args):

    # Load the synset words
    idx2cls = list()
    with open('samples/synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            idx2cls.append(line)

    # Setup a classification model
    print('Loading a model...', end='')
    model = torchvision.models.resnet152(pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    print('finished')

    # Setup the SmoothGrad
    smooth_grad = SmoothGrad(model=model, cuda=args.cuda, sigma=args.sigma,
                             n_samples=args.n_samples, guided=args.guided)

    # Predict without adding noises
    smooth_grad.load_image(filename=args.image, transform=transform)
    prob, idx = smooth_grad.forward()

    # Generate the saliency images of top 3
    for i in range(0, 3):
        print('{:.5f}\t{}'.format(prob[i], idx2cls[idx[i]]))
        smooth_grad.generate(
            filename='results/{}'.format(idx2cls[idx[i]]), idx=idx[i])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SmoothGrad visualization')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--sigma', type=float, default=0.20)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--guided', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
