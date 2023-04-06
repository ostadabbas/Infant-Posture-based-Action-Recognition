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

spinpose_outputs = '/work/aclab/xiaof.huang/fu.n/SPIN_infant_new/logs/infant_camL200_ft_e640_videos'
tar_dir = '/work/aclab/xiaof.huang/fu.n/InfantAction/pose_3d_res'
vis_dir = '/work/aclab/xiaof.huang/fu.n/InfantAction/pose_3d_vis'

frame_list = os.listdir(spinpose_outputs)
print(frame_list)

for i in range(len(frame_list)):
    vid_name = frame_list[i]
    res1_file = spinpose_outputs + '/' + vid_name + '/output_gt_3D.npy'
    res2_file = spinpose_outputs + '/' + vid_name + '/output_imgnames.npy'
    res3_file = spinpose_outputs + '/' + vid_name + '/output_pose_3D.npy'
    res4_file = spinpose_outputs + '/' + vid_name + '/eval_rst.npz'

    vid_dir = os.path.join(tar_dir, vid_name)
    os.mkdir(vid_dir)
    new1_file = vid_dir + '/output_gt_3D.npy'
    new2_file = vid_dir + '/output_imgnames.npy'
    new3_file = vid_dir + '/output_pose_3D.npy'
    new4_file = vid_dir + '/eval_rst.npz'
    shutil.copyfile(res1_file, new1_file)
    shutil.copyfile(res2_file, new2_file)
    shutil.copyfile(res3_file, new3_file)
    shutil.copyfile(res4_file, new4_file)

    new_vis_dir = os.path.join(vis_dir, vid_name)
    os.mkdir(new_vis_dir)

    vis_folder = spinpose_outputs + '/' + vid_name + '/vis'
    vis_list = os.listdir(vis_folder)
    for j in range(int(len(vis_list)/7)):
        vis_file = vis_folder + '/' + f'{j:05d}' + '_f_RGB.jpg'
        new_file = new_vis_dir + '/test' + str(j) + '.jpg'
        shutil.copyfile(vis_file, new_file)
