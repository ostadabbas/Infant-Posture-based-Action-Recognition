import os
import copy
import numpy as np
import json
#import h5py
from PIL import Image
import glob
import pickle
from shutil import copyfile
import cv2
import json
import sys
import re
import shutil

bbox_path = '/scratch/fu.n/InfantAction/bbox_data'

my_list = os.listdir(bbox_path)
for i in range(len(my_list)):
    path = os.path.join(bbox_path, my_list[i])
    output_fd = os.path.join(path, 'output')
    allfiles = os.listdir(output_fd)
    files = [ fname for fname in allfiles if fname.endswith('.png')]
    num = len(files)

    ori_bbox = os.path.join(path, 'results.json')
    f = open(ori_bbox)
    bbox_data = json.load(f)

    print('##########Processing Video ' + str(i+1))
    print(path)
    total = 0
    for j in range(num):
        name = 'frame'+str(j)
        if (name in bbox_data) and bbox_data[name][4] >= 0.6:
            total = total + 1

    print('Corrected BBox Number is ' + str(total))
    print('Total BBox Number is ' + str(num))


