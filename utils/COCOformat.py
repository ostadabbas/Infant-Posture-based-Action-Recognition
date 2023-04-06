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

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

root_path = '/home/faye/Documents/InfantProject'
img_dir = 'data/example3'
img_path = os.path.join(root_path, img_dir)

raw_file = 'outputs/example3_outputs/yolov3_example3_results.json'
bbox_path = os.path.join(root_path, raw_file)

new_img_dir = 'data/new_example3'
new_img_path = os.path.join(root_path, new_img_dir)
if not os.path.exists(new_img_path):
    os.mkdir(new_img_path)


tar_file = 'outputs/example3_outputs/person_keypoints_validate_example3.json'
tar_path = os.path.join(root_path, tar_file)

# process bar
def process_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
 
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
 
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
 
#root_path = "data_with_box_example/"
images, categories, annotations = [], [], []
 
categories.append({"supercategory": "person", "id": 1, "name": "person", "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]})

f = open(bbox_path,)
bbox_data = json.load(f)
print(len(bbox_data))
idx = 0

count = 1
total = 100 
for x in range(len(bbox_data)):
    #if x <= 99:
       #old_name = 'google-' + format(x, '02d') + '.png'
    #else:
       #old_name = 'youtube-' + format(x-100, '02d') + '.png'
    old_name = 'frame' + str(x) + '.jpg'
# for infile in sorted(glob.glob(img_path + '/*.jpg'), key=numericalSort): 
    process_bar(count, total)
    count += 1
    infile = os.path.join(img_path, old_name)
    # old_name = os.path.basename(infile)
    new_name = 'test' + str(idx) + '.jpg'
    new_name_path = os.path.join(new_img_path, new_name)
    #shutil.copy(infile, new_name_path)

    img_cv2 = cv2.imread(infile)
    cv2.imwrite(new_name_path, img_cv2)
    [height, width, _] = img_cv2.shape
    # images info
    images.append({"file_name": new_name, "is_synthetic": False, "frame_id": idx, "height": height, "width": width, "id": idx, "is_labeled": True, "nframes": 18000, "original_file_name": old_name, "posture": ""})
 
    """
    annotation info:
    id : anno_id_count
    category_id : category_id
    bbox : bbox
    segmentation : [segment]
    area : area
    iscrowd : 0
    image_id : image_id
    """
    category_id = 1
     
    old_name = old_name[:-4]
    if old_name in bbox_data:
        bbox = [bbox_data[old_name][0], bbox_data[old_name][1], bbox_data[old_name][2], bbox_data[old_name][3]]
        score = [bbox_data[old_name][4]]
    else:
        bbox = [float(0), float(0), float(width), float(height)]
        score = []

    area = width * height

    anno_info = {"bbox": bbox, "bbox_head": [], "category_id": 1, "id": idx, "image_id": idx, "keypoints": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], "scores": score, "track_id": 0, "num_keypoints": 17, "segmentation": [], "area": area, "iscrowd": 0} 
    annotations.append(anno_info)

    idx += 1
 
all_json = {"images": images, "annotations": annotations, "categories": categories}
 
with open(tar_path, "w") as outfile:
    json.dump(all_json, outfile)

