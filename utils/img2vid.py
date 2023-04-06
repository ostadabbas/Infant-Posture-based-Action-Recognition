import numpy as np
import glob
import cv2
import os
 
img_folder = '/home/faye/Documents/InfantProject/outputs/example3_outputs/posture_vis_raw'
imgfile_array = []
for filename in glob.glob(img_folder + '/*.jpg'):
    imgfile_array.append(filename)

img_array = []
num_frames = len(imgfile_array)
print(num_frames)
for i in range(num_frames):
    filename = 'test' + str(i) + '.jpg'
    img = cv2.imread(os.path.join(img_folder, filename))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    print(len(img_array))
 
 
out = cv2.VideoWriter('/home/faye/Documents/InfantProject/outputs/example3_outputs/example3_pose_posture_vis_raw.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
