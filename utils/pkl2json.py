import os
import copy
import numpy as np
import json
import h5py
from PIL import Image
import glob
import pickle
from shutil import copyfile

#######please add path to the sample json file here
json_file = open('./person_keypoints_train2017.json')
sample_data = json.load(json_file)


def getName(num):
    num = str(num)
    while len(num) < 3:
        num = '0' + num
    return num + '.jpg'

seen_vid = []
frame_id = 0

num = 0

pkl_folder = './pkl_files'
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(pkl_folder):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
for file in listOfFiles:
    print(file)
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(data)
    frames_kpts = data['all_keyps'][1]
    frames_boxes = data['all_boxes']
    frames_syn = data['synthetic']
    frames_img = data['images']
    print(frames_img)

    for i in range(len(frames_img)):
        print(frames_img[i])
        print(frames_img[i][3:-4])
        if frames_img[i][:3] == 'rea':
            tmp = int(frames_img[i][4:-4])
            print(tmp)
            file_name = 'train' + '%05d.jpg' % (tmp + 1)
        elif frames_img[i][:3] == 'syn':
            tmp = int(frames_img[i][3:-4])
            print(tmp)
            file_name = 'train1' + '%04d.jpg' % (tmp + 1)
        else:
            file_name = frames_img[i]
        print(file_name)
        img_data["file_name"] = file_name
        img_data['original_file_name'] = file_name
        if frames_syn[i] == 1:
            img_data['is_synthetic'] = True
        else:
            img_data['is_synthetic'] = False

        im = Image.open(os.path.join('pkl_imgs', file_name))
        width, height = im.size
        img_data['height'] = height
        img_data['width'] = width

        frame_id = 0
        print('test')
        print(file_name)

        num = int(file_name[5:-4])
        #print(num)
        img_data["frame_id"] = frame_id
        img_data["id"] = num

        final_data['images'].append(copy.deepcopy(img_data))

        body_box = [frames_boxes[i][0][0], frames_boxes[i][0][1], frames_boxes[i][0][2] - frames_boxes[i][0][0], frames_boxes[i][0][3] - frames_boxes[i][0][1]]
        area = body_box[2] * body_box[3]

        keypoints = []
        for nk in range(17):
            keypoints.append(frames_kpts[i][0][0][nk])
            keypoints.append(frames_kpts[i][0][1][nk])
            keypoints.append(frames_kpts[i][0][-1][nk])

        ann_data['bbox'] = body_box
        # ann_data['bbox_head'] = body_box
        ann_data['area'] = area
        ann_data['id'] = num
        ann_data['image_id'] = num
        ann_data['keypoints'] = keypoints

        final_data['annotations'].append(copy.deepcopy(ann_data))

        #num = num + 1


final_data['categories'] = sample_data[u'categories']
print(len(final_data['images']))

#### add path to where you want the final annotation file to be saved in
with open('./person_keypoints_train_infant.json', 'w') as fp:
    json.dump(final_data, fp)


