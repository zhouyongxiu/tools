# coding=utf-8
import os
import cv2
import random
import numpy as np
from imgaug import augmenters as iaa
import sys
import shutil

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

def img_aug(image):

    result = image

    case = random.randint(0, 4)
    # case = 4
    if (case == 0):
        q = random.randint(5, 20)
        cv2.imwrite('1.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        result = cv2.imread('1.jpg')

    elif (case == 1):
        blurer = iaa.MedianBlur(k=(3, 5))
        result = blurer.augment_image(result)  # blur image 2 by a sigma of 3.0

        blurer = iaa.ElasticTransformation(alpha=(0.5, 1.0), sigma=(0.1, 0.2))
        result = blurer.augment_image(result)  # blur image 3 by a sigma of 3.0 too

    elif (case == 2):
        q = random.randint(5, 20)
        cv2.imwrite('1.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        result = cv2.imread('1.jpg')

        blurer = iaa.MedianBlur(k=(3, 5))
        result = blurer.augment_image(result)  # blur image 2 by a sigma of 3.0

        blurer = iaa.ElasticTransformation(alpha=(0.5, 1.0), sigma=(0.1, 0.2))
        result = blurer.augment_image(result)  # blur image 3 by a sigma of 3.0 too

    elif (case == 3):

        blurer = iaa.MedianBlur(k=(3, 5))
        result = blurer.augment_image(result)  # blur image 2 by a sigma of 3.0

        blurer = iaa.ElasticTransformation(alpha=(0.5, 1.0), sigma=(0.1, 0.2))
        result = blurer.augment_image(result)  # blur image 3 by a sigma of 3.0

        q = random.randint(5, 20)
        cv2.imwrite('1.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        result = cv2.imread('1.jpg')

    elif (case == 4):

        x = random.randint(-8, 8)
        y = random.randint(-8, 8)
        while (x==0 and y==0):
            x = random.randint(-8, 8)
            y = random.randint(-8, 8)
        M = np.float32([[1, 0, x], [0, 1, y]])  # 10
        result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))  # 11
        #
        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        result = cv2.addWeighted(image, alpha, result, beta, gamma)

    return result



if __name__ == "__main__":

    root = '../data/mixture/train'
    aug = '../data/mixture/train_aug'

    max_num = 600 - 150

    # file_list = [f for f in os.listdir(root) if not f.startswith('.')]
    # input_list = ['侧脸_低头', '遮挡', '正脸']
    input_list = ['正脸']
    aug_file = '模糊'
    aug_path = os.path.join(aug, aug_file)

    if not os.path.exists(aug_path):
        os.mkdir(aug_path)

    input_path_list = []
    for input_file in input_list:
        img_name_list = [f for f in os.listdir(os.path.join(root, input_file)) if not f.startswith('.')]
        for img_name in img_name_list:
            input_path_list.append(os.path.join(root, input_file, img_name))
    random.shuffle(input_path_list)
    batch = len(input_path_list)
    count = 0
    while count < max_num:
        index = count % batch
        imagepath = input_path_list[index]
        img = cv2.imread(imagepath)
        # cv2.imshow('1', img)
        # img2 = img_aug(img)
        # cv2.imshow('2', img2)
        # cv2.waitKey()
        new_img = img_aug(img)
        output_path = os.path.join(aug_path, 'aug%d.jpg' % count)
        cv2.imwrite(output_path, new_img)
        count += 1

