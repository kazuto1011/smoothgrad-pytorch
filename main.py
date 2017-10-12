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
    smooth_grad = SmoothGrad(model=model, cuda=args.cuda,
                             guidedbackprop=args.guidedbp)
    smooth_grad.load_image(filename=args.image, transform=transform)
    smooth_grad.forward()
    idx = smooth_grad.idx
    prob = smooth_grad.prob

    # Generate the saliency images of top 3 classes
    for i in range(0, 3):
        print('{:.5f}\t{}'.format(prob[i], classes[idx[i]]))
        filename = 'results/{}'.format(classes[idx[i]])
        img = smooth_grad.generate(filename=filename, idx=idx[i])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SmoothGrad visualization')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--guidedbp', action='store_true', default=False)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
