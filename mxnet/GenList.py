# coding=utf-8
import os
import cv2
import random
import numpy
import imghdr

import sys

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# if __name__ == "__main__":

dict = {'wp': 0,
        'zc': 1,
        'bfzd': 2,
        'wqzd': 3}

rate = 0.1

root = unicode("./data/zd", "utf-8")

Trainlist = []
Testlist = []
index = 0
max_num = 80000

for types in dict:
    img_list = [f for f in os.listdir(os.path.join(root, types)) if not f.startswith('.')]
    total = len(img_list)
    # for i, img in enumerate(img_list):
    i = 0
    while (i < max_num):
        img = img_list[i % total]
        str0 = '%d\t%f\t%s\n' % (index, float(dict[types]), os.path.join(root, types, img))
        if i < int((1 - rate) * max_num):
            Trainlist.append(str0)
        else:
            Testlist.append(str0)
        index += 1

random.seed(100)
random.shuffle(Trainlist)
random.seed(200)
random.shuffle(Testlist)

Trainfile = open("./data/train1.lst", "w")
for str1 in Trainlist:
    Trainfile.write(str1.encode("utf8"))
Trainfile.close()

Testfile = open("./data/valid1.lst", "w")
for str1 in Testlist:
    Testfile.write(str1.encode("utf8"))
Testfile.close()


