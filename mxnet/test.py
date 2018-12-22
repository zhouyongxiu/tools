# -*- coding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import mxnet as mx
import cv2
import numpy as np
import time
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(fname):
    # download and show the image
    img = cv2.cvtColor(fname, cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

if __name__=="__main__":

    sym, arg_params, aux_params = mx.model.load_checkpoint('model1/plate', 1)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(3), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
             label_shapes=mod._label_shapes)
    print mod._label_shapes
    mod.set_params(arg_params, aux_params, allow_missing=True)
    with open('model1/label.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    root = './data/test/'
    output = './data/test/result'
    if not os.path.exists(output):
        os.makedirs(output)
    # result = './data/sample/result'
    # folders = ['attack', 'real']
    dict = {'未悬挂': 0,
            '正常': 1,
            '部分遮挡': 2,
            '完全遮挡': 3}
    # T = 84
    conut = 0
    namelist = [f for f in os.listdir(root) if not f.startswith('.')]
    with open('result.txt', 'w') as f:
        for img in namelist:
            time_st = time.time()
            img_path = os.path.join(root, img)
            image = cv2.imread(img_path)
            img_batch = get_image(image)
            # compute the predict probabilities
            mod.forward(Batch([mx.nd.array(img_batch)]))
            prob = mod.get_outputs()[0].asnumpy()
            # print the top-5
            prob = np.squeeze(prob)
            result = np.argsort(prob)[::-1]
            # for i in result[0:5]:
            #     print('probability=%f, class=%s' % (prob[i], labels[i]))

            # print sim
            label = labels[result[0]]
            conut += 1
            print conut
            time_ed = time.time()
            print('process time:%fms' % (time_ed * 1000 - time_st * 1000))
            f.write('%d#%s\n'%(result[0], img))
            # cv2.imwrite(os.path.join(result, '%s_' % label + img), image)
            # cv2.imwrite(os.path.join(result, img), tmp)
            # print os.path.join(result, folder, ('%03f_' % (sim)) + img)
            # shutil.copy(img_path, os.path.join(result, folder, ('%03f_' % (sim)) + img))
