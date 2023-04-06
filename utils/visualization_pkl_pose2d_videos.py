# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
import pickle

edges = [[5, 6], [5, 7],
          [7, 9], [6, 8], [8, 10], [11, 12], [11, 13], [13, 15],
          [12, 14], [14,16]]  # 17 keypoints format
         
#edges = [[0, 1], [1, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 7],
#         [1, 8], [8, 9], [8, 10], [10, 12], [9, 11], [11, 13],
#        [12, 14]]  # 15 keypoints format
edges = np.array(edges)

def generate_color_map():
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 12)]
    color_map = []
    for c in colors:
        b = max(0, min(255, int(np.floor(c[0] * 256.0))))
        g = max(0, min(255, int(np.floor(c[1] * 256.0))))
        r = max(0, min(255, int(np.floor(c[2] * 256.0))))
        color_map.append((b, g, r))
    return color_map

def generate_skel(img, kpts, edges, color_map):
    num_joints = kpts.shape[0]
    num_edge = len(edges)
    #plt.imshow(img)

    for i in range(num_edge):

        if kpts[edges[i, 0], 2] > 0.0 and kpts[edges[i, 1], 2] > 0.0:
            x0 = int(kpts[edges[i, 0], 0])
            x1 = int(kpts[edges[i, 1], 0])

            y0 = int(kpts[edges[i, 0], 1])
            y1 = int(kpts[edges[i, 1], 1])

            img = cv2.line(img, (x0, y0), (x1, y1), color = color_map[i], thickness = 6)

    if kpts[5, 2] > 0.0 and kpts[6, 2] > 0.0 and kpts[11, 2] > 0.5 and kpts[12, 2] > 0.0:
        x0 = int((kpts[5, 0] + kpts[6, 0]) / 2)
        x1 = int((kpts[11, 0] + kpts[12, 0]) / 2)

        y0 = int((kpts[5, 1] + kpts[6, 1]) / 2)
        y1 = int((kpts[11, 1] + kpts[12, 1]) / 2)

        img = cv2.line(img, (x0, y0), (x1, y1), color=color_map[i], thickness=6)

    for a in range(num_joints-5):
        a += 5
        if kpts[a, 2] > 0.0:
            img = cv2.circle(img, (int(kpts[a, 0]), int(kpts[a, 1])), 6, color_map[a - 5], -1)
   
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    # general
    '''
    parser.add_argument('--image-path',
                        help='Path of COCO val images',
                        type=str,
                        default='data/coco/images/val2017/'
                        )

    parser.add_argument('--gt-anno',
                        help='Path of COCO val annotation',
                        type=str,
                        default='data/coco/annotations/person_keypoints_val2017.json'
                        )
    '''
    parser.add_argument('--save-path',
                        help="Path to save the visualizations",
                        type=str,
                        default='visualization/coco/')

    parser.add_argument('--prediction',
                        help="Prediction file to visualize",
                        type=str,
                        required=True)

    args = parser.parse_args()

    return args


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict

def plot(data, save_path, img_dir, save=True):
    for i in range(len(data)):
        img_id = data[i]['image_id']
        img_file = os.path.join(img_dir, 'test'+str(img_id)+'.jpg')
        img = cv2.imread(img_file)
   
        tmps = data[i]['keypoints']
        kpts = np.zeros((17, 3))
        kpts[:,0] = tmps[0::3]
        kpts[:,1] = tmps[1::3]
        kpts[:,2] = tmps[2::3]

        img = generate_skel(img, kpts, edges, color_map)
        filename = os.path.basename(img_file)        
        output_path = os.path.join(save_path, filename) 
        cv2.imwrite(output_path, img) 
        #cv2.imshow(filename, img)
        #cv2.waitKey(0)

if __name__ == '__main__':
    save_path = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos8_demo_outputs/pose_2d_vis/new_1'
    #pred_2d_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/pose_2d_res'
    #pred_3d_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/pose_3d_res'
    #vid_list = os.listdir(pred_2d_root)
    #print(vid_list)
    
    img_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos8_demo_outputs/custom_syrip/1/images/validate_infant'
    pkl_file = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos8_demo_outputs/selected_gt.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    print(data)
    frames_kpts = data['all_keyps'][1]
    #frames_boxes = data['all_boxes']
    #frames_syn = data['synthetic']
    frames_img = data['images']
    print(frames_img)


    color_map = generate_color_map()

    for i in range(len(frames_img)):
        img_name = frames_img[i]
        img_file = os.path.join(img_root, 'test'+str(int(img_name[5:-4]))+'.jpg')
        img = cv2.imread(img_file)

        tmps = []
        for nk in range(17):
            tmps.append(frames_kpts[i][0][0][nk])
            tmps.append(frames_kpts[i][0][1][nk])
            tmps.append(frames_kpts[i][0][-1][nk])

        kpts = np.zeros((17, 3))
        kpts[:,0] = tmps[0::3]
        kpts[:,1] = tmps[1::3]
        kpts[:,2] = tmps[2::3]

        img = generate_skel(img, kpts, edges, color_map)
        filename = os.path.basename(img_file)        
        output_path = os.path.join(save_path, filename) 
        cv2.imwrite(output_path, img) 
        #cv2.imshow(filename, img)
        #cv2.waitKey(0)


