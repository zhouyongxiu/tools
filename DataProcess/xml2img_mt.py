# -*- coding: utf-8 -*-
import xml.dom.minidom
import sys
import os
import cv2
import math
from collections import namedtuple
# Box = namedtuple('Box', ['xmin', 'ymin', 'xmax', 'ymax'])
Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])
Obj = namedtuple('Obj', ['rect', 'manned'])
from multiprocessing import Pool


def img_crop(image, rect):

    pad = 0.1
    minx = int(rect.x - pad * rect.w)
    if minx < 0:
        minx = 0
    miny = int(rect.y - pad * rect.h)
    if miny < 0:
        miny = 0
    maxx = int(rect.x + rect.w + pad * rect.w)
    if maxx > image.shape[1]:
        maxx = image.shape[1]
    maxy = int(rect.y + rect.h + pad * rect.h)
    if maxy > image.shape[0]:
        maxy = image.shape[0]
    img_crop = image[miny:maxy, minx:maxx]
    return img_crop


def xml2obj(xml_name):

    result = []
    if not os.path.exists(xml_name):
        print 'no file: %s' % xml_name
        return result
    dom = xml.dom.minidom.parse(xml_name)
    root = dom.documentElement
    objects = root.getElementsByTagName('object')
    # obj_num = len(objects)
    for index, obj in enumerate(objects):
        name = obj.getElementsByTagName('name')[0].firstChild.data
        bndbox = obj.getElementsByTagName('bndbox')[0]
        x0 = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
        y0 = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
        w = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        h = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        box = Rect(x0,y0,w,h)
        # angle = float(bndbox.getElementsByTagName('angle')[0].firstChild.data)
        manned = int(obj.getElementsByTagName('manned')[0].firstChild.data)
        # spray = int(obj.getElementsByTagName('spray')[0].firstChild.data)
        obj = Obj(box, manned)
        result.append(obj)
    return result

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

def folder_aug(anno_dir, img_dir, manned_dir, lists):

    for xml_name in lists:
        xml_path = os.path.join(anno_dir, xml_name)
        objs = xml2obj(xml_path)
        img = os.path.join(img_dir, xml_name.split('.xml')[0] + '.jpg')
        if os.path.exists(img) and len(objs):
            image = cv2.imread(img)
            for i, obj in enumerate(objs):
                # print ('cover=%d' % (obj.cover))
                # print ('spray=%d' % (obj.spray))
                # rect = Rect(obj.box.xmin, obj.box.ymin, obj.box.xmax - obj.box.xmin, obj.box.ymax - obj.box.ymin)
                car = img_crop(image, obj.rect)
                cv2.imwrite(os.path.join(manned_dir, '%d' % obj.manned, xml_name.split('.xml')[0] + '_%d.jpg'%(i)), car)

if __name__ == "__main__":

    anno_dir = './data/Annotations'
    img_dir = './data/JPEGImages'
    manned_dir = './data/result_manned'
    if not os.path.exists(manned_dir):
        os.makedirs(manned_dir)
        os.makedirs(os.path.join(manned_dir, '0'))
        os.makedirs(os.path.join(manned_dir, '1'))

    thread_num = 10
    xml_list = [f for f in os.listdir(anno_dir) if not f.startswith('.')]
    lists = split_list(xml_list, thread_num)

    p = Pool(thread_num)
    for i in range(thread_num):
        p.apply_async(folder_aug, args=(anno_dir, img_dir, manned_dir, lists[i]))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')