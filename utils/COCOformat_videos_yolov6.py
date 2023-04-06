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

bboxes_data = '/work/aclab/xiaof.huang/fu.n/InfantAction/bbox_data'
frame_data = '/work/aclab/xiaof.huang/InfantActionData/new_videos'
custom_data = '/work/aclab/xiaof.huang/fu.n/InfantAction/custom_syrip'

frame_list = os.listdir(frame_data)
print(frame_list)
bbox_list = os.listdir(bboxes_data)
print(len(bbox_list))
for i in range(len(frame_list)):
    print(frame_list[i])
    bbox_folder = os.path.join(bboxes_data, frame_list[i]+'/'+frame_list[i])
    bbox_path = os.path.join(bbox_folder, 'results.json')
    img_path = os.path.join(frame_data, frame_list[i]) 
    my_list = os.listdir(img_path)
    print(img_path)
    num_frames = len(my_list)

    vid_folder = os.path.join(custom_data, frame_list[i])
    if not os.path.isdir(vid_folder):
        os.mkdir(vid_folder)

    anno_path = os.path.join(vid_folder, 'annotations')
    os.mkdir(anno_path)
    imgs_path = os.path.join(vid_folder, 'images')
    os.mkdir(imgs_path)
    validate_path = os.path.join(imgs_path, 'validate_infant')
    os.mkdir(validate_path)

    tar_file = os.path.join(anno_path, 'person_keypoints_validate_infant.json')

    # process bar
    def process_bar(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))
 
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
 
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()
 
    images, categories, annotations = [], [], []
 
    categories.append({"supercategory": "person", "id": 1, "name": "person", "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]})
    '''
    f = open(bbox_path,)
    bbox_data = json.load(f)
    print(len(bbox_data))
    '''
    idx = 0

    count = 1
    total = 100
    print(num_frames) 
    for x in range(num_frames):
        len_str = '04d'
        '''
        #if frame_list[i] == '112' or frame_list[i] == '113' or frame_list[i] == '128' or frame_list[i] == '129':
        #    len_str = '02d'
        if len(str(num_frames)) >= 4: 
            len_str = '0' + str(len(str(num_frames))) + 'd'  #'04d'
        len_str = '0' + str(len(str(num_frames))) + 'd'
        if frame_list[i] == '16' or frame_list[i] == '43' or frame_list[i] == '18' or frame_list[i] == '19' or frame_list[i] == '4'  or frame_list[i] == '41' or frame_list[i] == '3' or frame_list[i] == '199'  or frame_list[i] == '27' or frame_list[i] == '2' or frame_list[i] == '23' or frame_list[i] == '24':
            len_str = '03d'
        if frame_list[i] == '184':
            len_str = '03d'
            if len(str(x)) >= 4: 
                len_str = '0' + str(len(str(num_frames))) + 'd'  #'04d'
        if int(frame_list[i]) < 100:
            len_str = '03d'
        '''

        old_name = 'frame' + format(x, len_str) + '.jpg'
        process_bar(count, total)
        count += 1
        infile = os.path.join(img_path, old_name)
        new_name = 'test' + str(idx) + '.jpg'
        new_name_path = os.path.join(validate_path, new_name)
        print(infile)
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
        bbox_path = os.path.join(bbox_folder, old_name+'.txt')
        if os.path.exists(bbox_path):
            with open(os.path.join(bbox_folder, old_name+'.txt')) as bbox_f:
                bbox_data = [(line.strip()) for line in bbox_f.readlines()]
            #print(bbox_data)
            x = 0
            y = 0
            box_w = 0
            box_h = 0
            box_score = 0
            dist = np.sqrt(pow(width/2, 2)+pow(height/2, 2))
            for k in range(len(bbox_data)):
                bbox_list = [float(i) for i in bbox_data[k].split()]            
                if np.sqrt(pow(((bbox_list[1]+bbox_list[3])/2-width/2), 2)+pow(((bbox_list[2]+bbox_list[4])/2-height/2), 2)) < dist:
                    x = bbox_list[1]
                    y = bbox_list[2]
                    box_w = bbox_list[3] - bbox_list[1]
                    box_h = bbox_list[4] - bbox_list[2]
                    box_score = bbox_list[5]
                    if frame_list[i] == '180':
                        print(bbox_list)
            
            bbox = [x, y, box_w , box_h]
            score = [box_score]
        else:
            bbox = [float(0), float(0), float(width), float(height)]
            score = []

        area = width * height

        anno_info = {"bbox": bbox, "bbox_head": [], "category_id": 1, "id": idx, "image_id": idx, "keypoints": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], "scores": score, "track_id": 0, "num_keypoints": 17, "segmentation": [], "area": area, "iscrowd": 0} 
        annotations.append(anno_info)

        idx += 1
 
    all_json = {"images": images, "annotations": annotations, "categories": categories}
 
    with open(tar_file, "w") as outfile:
        json.dump(all_json, outfile)

