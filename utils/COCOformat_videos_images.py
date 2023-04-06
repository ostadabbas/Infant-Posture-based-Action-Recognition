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
import numpy as np

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

train_vid = np.load('/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset/train_videos.npy')

test_vid = np.load('/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset/test_videos.npy')

rest_vid = np.load('/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset/rest_videos.npy')

train_vid = np.concatenate((train_vid, rest_vid), axis=0)

ori_anno_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/custom_syrip'
ori_2d_pose_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/pose_2d_res'
ori_3d_pose_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/pose_3d_res'

tar_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/posture_model_inputs'
fd_2d = 'test100_train300_2d'
fd_3d = 'test100_train300_3d'

images, categories, annotations = [], [], []
 
categories.append({"supercategory": "person", "id": 1, "name": "person", "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]})


# train set preparation
train_img_list = []
train_gt_list = []
train_pred_list = []

id = 0
len_str = '05d'
for i in range(train_vid.shape[0]):
    vid_name = str(int(train_vid[i][0]))
    ori_vid_fd = ori_anno_fd + '/' + vid_name

    frame1_idx = int(train_vid[i][4] / 2)
    frame2_idx = int(train_vid[i][6] / 2) + int(train_vid[i][4]) + int(train_vid[i][5])
    ori_frame1_name = 'test' + str(frame1_idx - 1) + '.jpg'
    ori_frame2_name = 'test' + str(frame2_idx - 1) + '.jpg'
    ori_frame1_file = os.path.join(ori_vid_fd, 'images/validate_infant/' + ori_frame1_name)
    ori_frame2_file = os.path.join(ori_vid_fd, 'images/validate_infant/' + ori_frame2_name)
    
    img_names = np.load(os.path.join(ori_3d_pose_fd, vid_name + '/output_imgnames.npy'))
    img_list = img_names.tolist()
    gt_3d = np.load(os.path.join(ori_3d_pose_fd, vid_name + '/output_gt_3D.npy'))
    gt_list = gt_3d.tolist()
    pred_3d = np.load(os.path.join(ori_3d_pose_fd, vid_name + '/output_pose_3D.npy'))
    pred_list = pred_3d.tolist()

    pose_2d_file = os.path.join(ori_2d_pose_fd, vid_name + '/keypoints_validate_infant_results_0.json')
    f = open(pose_2d_file,)
    pose_2d = json.load(f)
   
    # frame1
    id = id + 1        
    new_name = 'train' + format(id, len_str) + '.jpg'
    new_name_path = os.path.join(tar_fd, fd_2d + '/images/train300/' + new_name)
    img_cv2 = cv2.imread(ori_frame1_file)
    cv2.imwrite(new_name_path, img_cv2)
    [height, width, _] = img_cv2.shape

    train_img_list.append(new_name) 
    train_gt_list.append(gt_list[frame1_idx-1])
    train_pred_list.append(pred_list[frame1_idx-1])
   
    posture = ""
    posture1_label = int(train_vid[i][8])
    if posture1_label == 0:
       posture = 'Supine'
    elif posture1_label == 1:
       posture = 'Prone'
    elif posture1_label == 2:
       posture = 'Sitting'
    elif posture1_label == 3:
       posture = 'Standing'
    elif posture1_label == 4:
       posture = 'All Fours'
    else:
       print('ERROR')

    # images info
    images.append({"file_name": new_name, "is_synthetic": False, "frame_id": id-1, "height": height, "width": width, "id": id-1, "is_labeled": True, "nframes": 18000, "original_file_name": ori_frame1_name, "original_video_name": vid_name, "posture": posture})
 
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
 
    kpts = pose_2d[frame1_idx - 1]['keypoints']
    scale = pose_2d[frame1_idx - 1]['scale']
    score = pose_2d[frame1_idx - 1]['score']
    center = pose_2d[frame1_idx - 1]['center']
    for j in range(17):
        kpts[j*3+2] = 1.0

    '''
    old_name = old_name[:-4]
    if old_name in bbox_data:
        bbox = [bbox_data[old_name][0], bbox_data[old_name][1], bbox_data[old_name][2], bbox_data[old_name][3]]
        score = [bbox_data[old_name][4]]
    else:
        bbox = [float(0), float(0), float(width), float(height)]
        score = []

    area = width * height
    '''
    anno_info = {"bbox": [], "bbox_head": [], "category_id": 1, "id": id-1, "image_id": id-1, "keypoints": kpts, "scores": score, "track_id": 0, "num_keypoints": 17, "segmentation": [], "area": [], "iscrowd": 0} 
    annotations.append(anno_info)

    # frame2
    id = id + 1        
    new_name = 'train' + format(id, len_str) + '.jpg'
    new_name_path = os.path.join(tar_fd, fd_2d + '/images/train300/' + new_name)
    img_cv2 = cv2.imread(ori_frame2_file)
    cv2.imwrite(new_name_path, img_cv2)
    [height, width, _] = img_cv2.shape

    train_img_list.append(new_name) 
    train_gt_list.append(gt_list[frame2_idx-1])
    train_pred_list.append(pred_list[frame2_idx-1])

    posture = ""
    posture2_label = int(train_vid[i][9])
    if posture2_label == 0:
       posture = 'Supine'
    elif posture2_label == 1:
       posture = 'Prone'
    elif posture2_label == 2:
       posture = 'Sitting'
    elif posture2_label == 3:
       posture = 'Standing'
    elif posture2_label == 4:
       posture = 'All Fours'
    else:
       print('ERROR')

    # images info
    images.append({"file_name": new_name, "is_synthetic": False, "frame_id": id-1, "height": height, "width": width, "id": id-1, "is_labeled": True, "nframes": 18000, "original_file_name": ori_frame2_name, "original_video_name": vid_name, "posture": posture})
 
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

    kpts = pose_2d[frame2_idx - 1]['keypoints']
    scale = pose_2d[frame2_idx - 1]['scale']
    score = pose_2d[frame2_idx - 1]['score']
    center = pose_2d[frame2_idx - 1]['center']
    for j in range(17):
        kpts[j*3+2] = 1.0

    '''
    old_name = old_name[:-4]
    if old_name in bbox_data:
        bbox = [bbox_data[old_name][0], bbox_data[old_name][1], bbox_data[old_name][2], bbox_data[old_name][3]]
        score = [bbox_data[old_name][4]]
    else:
        bbox = [float(0), float(0), float(width), float(height)]
        score = []

    area = width * height
    '''
    anno_info = {"bbox": [], "bbox_head": [], "category_id": 1, "id": id-1, "image_id": id-1, "keypoints": kpts, "scores": score, "track_id": 0, "num_keypoints": 17, "segmentation": [], "area": [], "iscrowd": 0} 
    annotations.append(anno_info)

all_json = {"images": images, "annotations": annotations, "categories": categories}

with open(os.path.join(tar_fd, fd_2d + '/annotations/train300/person_keypoints_train_infant_vidframes_5class.json'), "w") as outfile:
    json.dump(all_json, outfile)

print(len(train_img_list))
with open(os.path.join(tar_fd, fd_3d + '/train300/output_imgnames.npy'), 'wb') as f:
    np.save(f, np.array(train_img_list))
with open(os.path.join(tar_fd, fd_3d + '/train300/output_gt_3D.npy'), 'wb') as f:
    np.save(f, np.array(train_gt_list))
with open(os.path.join(tar_fd, fd_3d + '/train300/output_pose_3D.npy'), 'wb') as f:
    np.save(f, np.array(train_pred_list))


all_json.clear()
print(all_json)

images, categories, annotations = [], [], []

# test set preparation
test_img_list = []
test_gt_list = []
test_pred_list = []

id = 0
#len_str = '05d'
for i in range(test_vid.shape[0]):
    vid_name = str(int(test_vid[i][0]))
    ori_vid_fd = ori_anno_fd + '/' + vid_name

    frame1_idx = int(test_vid[i][4] / 2)
    frame2_idx = int(test_vid[i][6] / 2) + int(test_vid[i][4]) + int(test_vid[i][5])
    ori_frame1_name = 'test' + str(frame1_idx - 1) + '.jpg'
    ori_frame2_name = 'test' + str(frame2_idx - 1) + '.jpg'
    ori_frame1_file = os.path.join(ori_vid_fd, 'images/validate_infant/' + ori_frame1_name)
    ori_frame2_file = os.path.join(ori_vid_fd, 'images/validate_infant/' + ori_frame2_name)
    
    img_names = np.load(os.path.join(ori_3d_pose_fd, vid_name + '/output_imgnames.npy'))
    img_list = img_names.tolist()
    gt_3d = np.load(os.path.join(ori_3d_pose_fd, vid_name + '/output_gt_3D.npy'))
    gt_list = gt_3d.tolist()
    pred_3d = np.load(os.path.join(ori_3d_pose_fd, vid_name + '/output_pose_3D.npy'))
    pred_list = pred_3d.tolist()

    pose_2d_file = os.path.join(ori_2d_pose_fd, vid_name + '/keypoints_validate_infant_results_0.json')
    f = open(pose_2d_file,)
    pose_2d = json.load(f)


    # frame1
    id = id + 1        
    new_name = 'test' + str(id-1) + '.jpg'
    new_name_path = os.path.join(tar_fd, fd_2d + '/images/test100/' + new_name)
    img_cv2 = cv2.imread(ori_frame1_file)
    cv2.imwrite(new_name_path, img_cv2)
    [height, width, _] = img_cv2.shape

    test_img_list.append(new_name) 
    test_gt_list.append(gt_list[frame1_idx-1])
    test_pred_list.append(pred_list[frame1_idx-1])

    posture = ""
    posture1_label = int(test_vid[i][8])
    if posture1_label == 0:
       posture = 'Supine'
    elif posture1_label == 1:
       posture = 'Prone'
    elif posture1_label == 2:
       posture = 'Sitting'
    elif posture1_label == 3:
       posture = 'Standing'
    elif posture1_label == 4:
       posture = 'All Fours'
    else:
       print('ERROR')

    # images info
    images.append({"file_name": new_name, "is_synthetic": False, "frame_id": id-1, "height": height, "width": width, "id": id-1, "is_labeled": True, "nframes": 18000, "original_file_name": ori_frame1_name, "original_video_name": vid_name, "posture": posture})
 
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

    kpts = pose_2d[frame1_idx - 1]['keypoints']
    scale = pose_2d[frame1_idx - 1]['scale']
    score = pose_2d[frame1_idx - 1]['score']
    center = pose_2d[frame1_idx - 1]['center']
    for j in range(17):
        kpts[j*3+2] = 1.0

    '''
    old_name = old_name[:-4]
    if old_name in bbox_data:
        bbox = [bbox_data[old_name][0], bbox_data[old_name][1], bbox_data[old_name][2], bbox_data[old_name][3]]
        score = [bbox_data[old_name][4]]
    else:
        bbox = [float(0), float(0), float(width), float(height)]
        score = []

    area = width * height
    '''
    anno_info = {"bbox": [], "bbox_head": [], "category_id": 1, "id": id-1, "image_id": id-1, "keypoints": kpts, "scores": score, "track_id": 0, "num_keypoints": 17, "segmentation": [], "area": [], "iscrowd": 0} 
    annotations.append(anno_info)

    # frame2
    id = id + 1        
    new_name = 'test' + str(id-1) + '.jpg'
    new_name_path = os.path.join(tar_fd, fd_2d + '/images/test100/' + new_name)
    img_cv2 = cv2.imread(ori_frame2_file)
    cv2.imwrite(new_name_path, img_cv2)
    [height, width, _] = img_cv2.shape

    test_img_list.append(new_name) 
    test_gt_list.append(gt_list[frame2_idx-1])
    test_pred_list.append(pred_list[frame2_idx-1])

    posture = ""
    posture2_label = int(test_vid[i][9])
    if posture2_label == 0:
       posture = 'Supine'
    elif posture2_label == 1:
       posture = 'Prone'
    elif posture2_label == 2:
       posture = 'Sitting'
    elif posture2_label == 3:
       posture = 'Standing'
    elif posture2_label == 4:
       posture = 'All Fours'
    else:
       print('ERROR')

    # images info
    images.append({"file_name": new_name, "is_synthetic": False, "frame_id": id-1, "height": height, "width": width, "id": id-1, "is_labeled": True, "nframes": 18000, "original_file_name": ori_frame2_name, "original_video_name": vid_name, "posture": posture})
 
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
 
    kpts = pose_2d[frame2_idx - 1]['keypoints']
    scale = pose_2d[frame2_idx - 1]['scale']
    score = pose_2d[frame2_idx - 1]['score']
    center = pose_2d[frame2_idx - 1]['center']
    for j in range(17):
        kpts[j*3+2] = 1.0

    '''
    old_name = old_name[:-4]
    if old_name in bbox_data:
        bbox = [bbox_data[old_name][0], bbox_data[old_name][1], bbox_data[old_name][2], bbox_data[old_name][3]]
        score = [bbox_data[old_name][4]]
    else:
        bbox = [float(0), float(0), float(width), float(height)]
        score = []

    area = width * height
    '''
    anno_info = {"bbox": [], "bbox_head": [], "category_id": 1, "id": id-1, "image_id": id-1, "keypoints": kpts, "scores": score, "track_id": 0, "num_keypoints": 17, "segmentation": [], "area": [], "iscrowd": 0} 
    annotations.append(anno_info)

all_json = {"images": images, "annotations": annotations, "categories": categories}

with open(os.path.join(tar_fd, fd_2d + '/annotations/test100/person_keypoints_validate_infant_vidframes_5class.json'), "w") as outfile:
    json.dump(all_json, outfile)

print(len(test_img_list))
with open(os.path.join(tar_fd, fd_3d + '/test100/output_imgnames.npy'), 'wb') as f:
    np.save(f, np.array(test_img_list))
with open(os.path.join(tar_fd, fd_3d + '/test100/output_gt_3D.npy'), 'wb') as f:
    np.save(f, np.array(test_gt_list))
with open(os.path.join(tar_fd, fd_3d + '/test100/output_pose_3D.npy'), 'wb') as f:
    np.save(f, np.array(test_pred_list))


