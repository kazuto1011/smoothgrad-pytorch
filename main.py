#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-11

from __future__ import print_function

import argparse

import cv2
import numpy as np
import torchvision
from torchvision import transforms

from smooth_grad import SmoothGrad


def main(args):

    # Load the synset words
    file_name = 'synset_words.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ', 1)[
                           1].split(', ', 1)[0].replace(' ', '_'))

    print('Loading a model...')
    model = torchvision.models.resnet152(pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    smooth_grad = SmoothGrad(model=model, n_class=1000, cuda=args.cuda, guidedbackprop=True)
    smooth_grad.load_image(filename=args.image, transform=transform)
    smooth_grad.forward()

    idx = smooth_grad.idx
    prob = smooth_grad.prob

    for i in range(0, 5):
        cls_name = classes[idx[i]]
        filename = 'results/guided/{}.png'.format(cls_name)
        img = smooth_grad.generate(filename=filename, idx=idx[i])
        print('\t{:.5f}\t{}'.format(prob[i], cls_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grad-CAM visualization')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
