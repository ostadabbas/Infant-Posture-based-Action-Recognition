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

darkpose_outputs = '/work/aclab/xiaof.huang/fu.n/DarkPose1/output_videos'
tar_dir = '/work/aclab/xiaof.huang/fu.n/InfantAction/pose_2d_res'

frame_list = os.listdir(darkpose_outputs)
print(frame_list)

for i in range(len(frame_list)):
    vid_name = frame_list[i]
    res_file = darkpose_outputs + '/' + vid_name + '/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_custom/results/keypoints_validate_infant_results_0.json'
    vid_dir = os.path.join(tar_dir, vid_name)
    os.mkdir(vid_dir)
    new_file = vid_dir + '/keypoints_validate_infant_results_0.json'
    shutil.copyfile(res_file, new_file)