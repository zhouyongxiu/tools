#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:40:50 2017

@author: zyx
"""

import cv2
import os
import time

def extract(root_path, result_path, frequency):
    
    if os.path.exists(result_path):
        print ('%s exists\t\n'%result_path)
        return 
    os.mkdir(result_path)
                    
    cap = cv2.VideoCapture(root_path)
    if not cap.isOpened():
        f = open('log.txt', 'w')
        f.write('open %s failed \t\n' % root_path)
        f.close()
        print ('open %s failed \t\n' % root_path)
        return -1
    
    success = True
    index = 0
    while 1:
        success, frame = cap.read()
        if not success:
            break
        if index % frequency == 0:
            cv2.imwrite(os.path.join(result_path, '%05d.jpg'%index), frame)
            pass
        index += 1
    cap.release()

if __name__ == "__main__":
    
    input_path = '/Users/zyx/Desktop/data/face'
    output_path = '/Users/zyx/Desktop/data/face_img'
    frequency = 40
    id_list = [f for f in os.listdir(input_path) if not f.startswith('.')]
    
    for id in id_list:
        output_id = os.path.join(output_path, id)
        if not os.path.exists(output_id):
            os.mkdir(output_id)
        for root, dirs, files in os.walk(os.path.join(input_path, id)):  
            for file in files:  
                if os.path.splitext(file)[1] == '.MP4' or os.path.splitext(file)[1] == '.mp4':
                    root_path = os.path.join(input_path, id, file)
                    result_path = os.path.join(output_id, os.path.splitext(file)[0])
                    print ('process %s\t\n'%root_path)
                    start = time.time()
                    extract(root_path, result_path, frequency) 
                    end = time.time()
                    print ('time = %f s\t\n'%(end - start))
                
    

        
