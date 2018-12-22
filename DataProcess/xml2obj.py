# -*- coding: utf-8 -*-
import xml.dom.minidom
import sys
import os
import cv2

from collections import namedtuple
Box = namedtuple('Box', ['xmin', 'ymin', 'xmax', 'ymax'])
Obj = namedtuple('Obj', ['box', 'cover', 'spray'])
Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])


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
        x1 = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        y1 = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        box = Box(x0,y0,x1,y1)
        # angle = float(bndbox.getElementsByTagName('angle')[0].firstChild.data)
        cover = int(obj.getElementsByTagName('cover')[0].firstChild.data)
        spray = int(obj.getElementsByTagName('spray')[0].firstChild.data)
        obj = Obj(box, cover, spray)
        result.append(obj)
    return result


if __name__ == "__main__":

    anno_dir = './data/Annotations'
    img_dir = './data/JPEGImages'
    result_dir = './data/result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    xml_list = [f for f in os.listdir(anno_dir) if not f.startswith('.')]
    for xml_name in xml_list:
        xml_path = os.path.join(anno_dir, xml_name)
        objs = xml2obj(xml_path)
        img = os.path.join(img_dir, xml_name.split('.xml')[0] + '.jpg')
        if os.path.exists(img) and len(objs):
            image = cv2.imread(img)
            for i, obj in enumerate(objs):
                print ('cover=%d' % (obj.cover))
                print ('spray=%d' % (obj.spray))
                rect = Rect(obj.box.xmin, obj.box.ymin, obj.box.xmax - obj.box.xmin, obj.box.ymax - obj.box.ymin)
                car =  img_crop(image, rect)
                cv2.imwrite(os.path.join(result_dir, xml_name.split('.xml')[0] + '_%d.jpg'%(i)), car)
                # cv2.imshow('test', car)
                # cv2.waitKey(0)
