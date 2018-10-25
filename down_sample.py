# coding=utf-8
import os
import cv2
import random
import numpy
from imgaug import augmenters as iaa
import sys
import shutil

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

def img_aug(image):

    rotate = iaa.Affine(rotate=(-5, 5), mode = ["edge"])
    result = rotate.augment_image(image)

    contrastnormalization = iaa.ContrastNormalization((0.9, 1.1))
    result = contrastnormalization.augment_image(result)

    grayscale = iaa.Grayscale(alpha=(0, 0.2), from_colorspace="BGR")
    result = grayscale.augment_image(result)

    # blurer = iaa.GaussianBlur((0.0, 1.0))
    # result = blurer.augment_image(result)  # blur image 2 by a sigma of 3.0
    #
    # if (random.randint(0, 9) == 0):
    #     salter = iaa.SaltAndPepper((0.0, 0.01))
    #     result = salter.augment_image(result)
    #
    # gauss = iaa.AdditiveGaussianNoise(scale=(0.0, 0.03 * 255), per_channel=0.5)
    # result = gauss.augment_image(result)  # blur image 3 by a sigma of 3.0 too
    #
    # add = iaa.Add((-30, 30), per_channel=0.3)
    # result = add.augment_image(result)

    return result



if __name__ == "__main__":

    root = '../data/mixture/train'
    aug = '../data/mixture/train_aug'

    max_num = 30000

    # file_list = [f for f in os.listdir(root) if not f.startswith('.')]
    # file_list = ['侧脸_低头', '遮挡_非脸', '正脸']
    file_list = ['模糊']

    for file in file_list:

        aug_file = os.path.join(aug, file)
        if not os.path.exists(aug_file):
            os.mkdir(aug_file)
        img_name_list = [f for f in os.listdir(os.path.join(root, file)) if not f.startswith('.')]
        random.shuffle(img_name_list)
        count = 0
        for img_name in img_name_list:
            print count
            if (count >= max_num):
                break
            input_img = os.path.join(root, file, img_name)
            if (img_name.endswith('.jpg')):
                output_img = os.path.join(aug_file, img_name.split('.jpg')[0] + '.jpg')
            elif (img_name.endswith('.png')):
                output_img = os.path.join(aug_file, img_name.split('.png')[0] + '.jpg')
            else:
                continue
            img = cv2.imread(input_img)
            cv2.imwrite(output_img, img)
            count += 1
            # shutil.copy(input_img, output_img)

