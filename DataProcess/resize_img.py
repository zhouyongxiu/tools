#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:40:50 2017

@author: zyx
"""

import cv2
import os
import time

if __name__ == '__main__':

    input_path = '../data/face_folder'
    output_path = '../data/resize_folder'
    h = 224
    w = 224
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    index = 0
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            if not filename.endswith('jpg'):
                continue
            # print(os.path.join(dirpath, filename))
            img_path = os.path.join(dirpath, filename)
            out_path = os.path.join(dirpath.replace(input_path, output_path), filename)
            if not os.path.exists(dirpath.replace(input_path, output_path)):
                os.makedirs(dirpath.replace(input_path, output_path))
            image = cv2.imread(img_path)
            if image.shape[0] == image.shape[1]:
                image = cv2.resize(image, (w, h))
            else:
                print (image.shape[0], image.shape[1])
                if image.shape[0] > image.shape[1]:
                    image = image[0:image.shape[1], :, :]
                else:
                    x = int((image.shape[1] - image.shape[0]) * 0.5)
                    image = image[:, x:x + image.shape[0], :]
                image = cv2.resize(image, (w, h))
            cv2.imwrite(out_path, image)
            index += 1
            print (index)