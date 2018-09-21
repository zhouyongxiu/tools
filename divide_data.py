# -*- coding: utf-8 -*-

import os
import shutil
import random

if __name__ == '__main__':

    org = './data/walmat/org'
    train = './data/walmat/train'
    valid = './data/walmat/valid'

    valid_num = 50

    file_list = [f for f in os.listdir(org) if not f.startswith('.')]

    for file in file_list:

        train_file = os.path.join(train, file)
        valid_file = os.path.join(valid, file)

        if not os.path.exists(train_file):
            os.mkdir(train_file)
        if not os.path.exists(valid_file):
            os.mkdir(valid_file)

        img_name_list = [f for f in os.listdir(os.path.join(org, file)) if not f.startswith('.')]
        random.shuffle(img_name_list)

        for i, img_name in enumerate(img_name_list):

            if i < valid_num:
                input_img = os.path.join(org, file,  img_name)
                output_img = os.path.join(valid_file, img_name)
                shutil.copy(input_img, output_img)
            else:
                input_img = os.path.join(org, file, img_name)
                output_img = os.path.join(train_file, img_name)
                shutil.copy(input_img, output_img)