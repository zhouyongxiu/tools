# coding=utf-8
import os
import cv2
import random
import numpy
from imgaug import augmenters as iaa
import sys
import math
import threading

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

def img_aug(image):

    rotale = iaa.Affine(rotate=(-5, 5), mode = ["edge"])
    result = rotale.augment_image(image)

    blurer = iaa.GaussianBlur((0.0, 1.0))
    result = blurer.augment_image(result)  # blur image 2 by a sigma of 3.0

    if (random.randint(0, 9) == 0):
        salter = iaa.SaltAndPepper((0.0, 0.01))
        result = salter.augment_image(result)

    gauss = iaa.AdditiveGaussianNoise(scale=(0.0, 0.03 * 255), per_channel=0.5)
    result = gauss.augment_image(result)  # blur image 3 by a sigma of 3.0 too

    add = iaa.Add((-30, 30), per_channel=0.3)
    result = add.augment_image(result)

    return result

def folder_aug(root, result_path, lists, max_num, thread_num):

    name_index = 0
    for name in lists:
        print ('thrend = %d, name_index = %d\n' % (thread_num, name_index))
        name_index += 1
        for root2, dirs2, imgs in os.walk(os.path.join(root, name)):
            batch = len(imgs)
            count = len(imgs)
            while count < max_num:
                index = count % batch
                imagepath = os.path.join(root, name, imgs[index])
                img = cv2.imread(imagepath.encode("utf8"))

                new_img = img_aug(img)

                cv2.imwrite('%s/%s/%s_%04d.jpg' % (result_path, name, name, count + 1000), new_img)
                count += 1

def split_list(name_list, thread_num):

    lists = [[] for i in range(thread_num)]
    count = len(name_list)
    batch_size = math.ceil(float(count) / thread_num)

    for i in range(thread_num - 1):
        start = i * batch_size
        end = start + batch_size
        lists[i] = name_list[int(start): int(end)]

    lists[thread_num - 1] = name_list[int(batch_size * (thread_num - 1)):]

    return lists

if __name__ == "__main__":

    thread_num = 30
    max_num = 100

    root = unicode("remove", "utf-8")
    result_path = 'aug'
    names = os.listdir(root)
    lists = split_list(names, thread_num)
    #print (lists)

    threads = []  # 定义一个线程列表
    for i in range(thread_num):
        threads.append(threading.Thread(target=folder_aug, args=(root, result_path, lists[i], max_num,i)))

    for t in threads:
        t.setDaemon(True)
        t.start()

    for t in threads:
        t.join()

    print ('done')

