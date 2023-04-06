# import required packages and global variables
import sys
import math
import copy
import numpy
import os
import pickle
import os.path as osp
import numpy as np
import cv2 as cv
import glob
from ParticleFilter import particle_filter

DTYPE = np.float32
height = 256  # frame height in pixel
width = 256  # frame width in pixel
fps = 30
col_ch = 3
sigma = 2
resize_scale = 1
crop = False
classes=[
    'Typical',
    'Atypical'
]
keypoints = [
        'nose',
        'neck',
        'top',
        'NA',
        'NA',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle']
        

def colorize(im, total_fr, fr_no, joint):
    n_ch = im.shape[0]
    im_colorized = im
    intensity = 1 + 1.*fr_no/total_fr
    if joint==0:
        oc_t = [1,0.1,0.1]
        for i in range(n_ch):
            im_colorized[i, :, :] = im[i, :, :] * oc_t[i] * intensity
    elif joint==1:
        oc_t = [0.1,1,0.1]
        for i in range(n_ch):
            im_colorized[i, :, :] = im[i, :, :] * oc_t[i] * intensity
    else:
        oc_t = [0.1,0.1,1]
        for i in range(n_ch):
            im_colorized[i, :, :] = im[i, :, :] * oc_t[i] * intensity
    return im_colorized, oc_t


def joint_im(im_size, ksize, kernel, fr_no=None, keyps=None, joint_name=None, joint_pos=None, kepoint_list=keypoints,pid=0):
    """
    constructs an pose map image for each body joint by creating a given kernel around the joint location.
    ====== input =======
    im_size: size of the desired pose map
    ksize: size of the kernel
    kernel: desired kernel to be used for the pose map generation
    keyps: location of the key point with respect to the given image size
    joint_name: name of the body joint
    joint_pos: position of the joint with respect to the given image size
    keypoint_list: the list containing the name of the body joints with the order of results in the keyps

    ======= output ========
    joint_im: pose map corresponding the given body joint
    """
    if joint_pos is None:
        if joint_name is not None and keyps is not None:
            joint_pos = keyps[fr_no][pid][:2, kepoint_list.index(joint_name)]
        else:
            print('you have to input either joint position or joint name')
            return

    joint_im = np.zeros(im_size, dtype=DTYPE)
    x1 = int(joint_pos[0] - ksize / 2.0)
    x2 = x1 + ksize
    y1 = int(joint_pos[1] - ksize / 2.0)
    y2 = y1 + ksize
    if x1 < 0:
        kernel = kernel[-x1:,:]
        x1 = 0
    if y1 < 0 :
        kernel = kernel[:,-y1:]
        y1 = 0
    try:
        joint_im[:, x1:x2, y1:y2] = kernel
    except Exception as e: 
        pass
    return joint_im
    


def PF(kps):
    keyps = np.zeros((len(kps), 4, 17))
    for k in range(len(kps)):
        if kps[k]!=[]:
            keyps[k] = kps[k][0]     
    ave = sum(keyps)[:2]/len(keyps)
    for j in [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16]:
        initial_x = list(ave[:,j])
        initial_x.append(0)
        keyps = particle_filter(keyps, j, initial_x)
    
    for k in range(len(kps)):
        kps[k][0] = keyps[k]
    return kps



def Interpolation(kps):
    keyps = np.zeros((len(kps), 4, 17))
    for k in range(len(kps)):
        if kps[k]!=[]:
            keyps[k] = kps[k][0]
    
    zeros = []
    nonzeros = []
    for i,k in enumerate(keyps):
        if np.sum(k)==0:
            zeros.append(i)
        else:
            nonzeros.append(i)
    
    if nonzeros[0]!=0:
        keyps[0:nonzeros[0]] = keyps[nonzeros[0]]
        for idx in range(nonzeros[0]):
            zeros.remove(idx)
            nonzeros.append(idx)
    
    zeros = sorted(zeros)
    nonzeros = sorted(nonzeros)
    
    if nonzeros[-1] != keyps.shape[0]-1:
        keyps[nonzeros[-1]:] = keyps[nonzeros[-1]]
        for idx in range(nonzeros[-1]+1, keyps.shape[0]):
            zeros.remove(idx)
            nonzeros.append(idx)

    zeros = sorted(zeros)
    nonzeros = sorted(nonzeros)

    for zid in zeros:
        start = zid-1
        end = zid
        count = 0
        while end not in nonzeros:
            end+=1
            count+=1
        keyps[zid] = keyps[start] + (keyps[end]-keyps[start])/count
        nonzeros.append(zid)
        
    for k in range(len(kps)):
        if len(kps[k])>0:
            kps[k][0] = keyps[k]
        else:
            kps[k].append(keyps[k])
    return kps


def getBackground(ch, img_size):
    return np.zeros((ch, img_size[0], img_size[1]))
    
    
    
def PoTion(in_pose, original_size, img_size=(height, width), ch=3, sigma=2, scale=1):
    """
    Recieves a pickle file storing the result of the pose estimation and target tracking.
    creates a Gaussian with the given variance around each joint to create pose maps.
    Then aggregates pose maps in consequent frames to a single image by time dependant color coding.
    ==== input ====
    in_pose: path to the pickle file of the pose estimation results for each action clip
    img_size : size of the original images which the pose estimation is executed on
    col_ch: number of channels for the colorization
    sigma: standard deviation of the Gaussian distrinution around each limb joint
    ==== output ====
    A multi channel image pose evolution representation of the action clip
    """
    img_size = (int(scale * img_size[0]), int(scale * img_size[1]))
    n_ch = 15 * ch
    eps = 1e-5
    pose_mo = np.zeros((n_ch, img_size[0], img_size[1]), dtype=DTYPE)
    try:
        with open(in_pose, 'rb') as res:
            dets = pickle.load(res, encoding = 'latin1')
    except:
        return None
        
    all_keyps = dets["all_keyps"]
    keyps = np.array(all_keyps[1])
    total_fr = len(keyps)
    if total_fr<60:
        return None
    
    
    n_missing = 0
    for i in range(total_fr):
        if np.sum(keyps[i])==0:
            n_missing+=1
            
    if n_missing>total_fr*0.9:
        return None
        
        
    original_h, original_w = original_size
    for i,k1 in enumerate(keyps):
        for j,k2 in enumerate(k1):
            keyps[i][j][0] = k2[0] / original_h * height
            keyps[i][j][1] = k2[1] / original_w * width
            
    if n_missing!=0:
        keyps = Interpolation(keyps)
        
    if n_missing>total_fr*0.1:
        keyps=PF(keyps)
     
    ksize = int(np.ceil(6 * sigma))
    nose_mo = getBackground(ch,img_size)
    neck_mo = getBackground(ch,img_size)
    top_mo = getBackground(ch,img_size)
    left_shoulder_mo = getBackground(ch,img_size)
    right_shoulder_mo = getBackground(ch,img_size)
    left_elbow_mo = getBackground(ch,img_size)
    right_elbow_mo = getBackground(ch,img_size)
    left_wrist_mo = getBackground(ch,img_size)
    right_wrist_mo = getBackground(ch,img_size)
    left_hip_mo = getBackground(ch,img_size)
    right_hip_mo = getBackground(ch,img_size)
    left_knee_mo = getBackground(ch,img_size)
    right_knee_mo = getBackground(ch,img_size)
    left_ankle_mo = getBackground(ch,img_size)
    right_ankle_mo = getBackground(ch,img_size)

    gauss_ker = cv.getGaussianKernel(ksize=ksize, sigma=sigma)
    gauss_ker = gauss_ker / max(gauss_ker)
    gauss_ker_2d = gauss_ker * gauss_ker.T
    
    x = 0
    for i in range(total_fr):
        if np.sum(keyps[i]) != 0:
            for j in range(len(keyps[i])):
                valid = 0
                for kid in [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16]:
                    if keyps[i][j][0][kid] < height and keyps[i][j][1][kid] < width:
                        valid += 1
                if valid >= 3:
                    ###### Nose
                    x+=1
                    nose_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                                keyps=keyps, joint_name='nose', fr_no=i, pid=j)
                    nose_im = colorize(nose_im, total_fr, fr_no=i, joint=0)
                    nose_mo = nose_mo + nose_im[0]

                    ###### Upper Neck
                    neck_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                                keyps=keyps, joint_name='neck', fr_no=i, pid=j)
                    neck_im = colorize(neck_im, total_fr, fr_no=i, joint=0)
                    neck_mo = neck_mo + neck_im[0]

                    ###### Head Top
                    top_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                                keyps=keyps, joint_name='top', fr_no=i, pid=j)
                    top_im = colorize(top_im, total_fr, fr_no=i, joint=0)
                    top_mo = top_mo + top_im[0]

                    ###### Left shoulder
                    left_shoulder_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                                keyps=keyps, joint_name='left_shoulder', fr_no=i, pid=j)
                    left_shoulder_im = colorize(left_shoulder_im, total_fr, fr_no=i, joint=1)
                    left_shoulder_mo = left_shoulder_mo + left_shoulder_im[0]

                    ###### Right shoulder
                    right_shoulder_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                                 keyps=keyps, joint_name='right_shoulder', fr_no=i, pid=j)
                    right_shoulder_im = colorize(right_shoulder_im, total_fr, fr_no=i, joint=1)
                    right_shoulder_mo = right_shoulder_mo + right_shoulder_im[0]

                    ###### Left elbow
                    left_elbow_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                             keyps=keyps, joint_name='left_elbow', fr_no=i, pid=j)
                    left_elbow_im = colorize(left_elbow_im, total_fr, fr_no=i, joint=1)
                    left_elbow_mo = left_elbow_mo + left_elbow_im[0]

                    ###### Right elbow
                    right_elbow_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                              keyps=keyps, joint_name='right_elbow', fr_no=i, pid=j)
                    right_elbow_im = colorize(right_elbow_im, total_fr, fr_no=i, joint=1)
                    right_elbow_mo = right_elbow_mo + right_elbow_im[0]

                    ###### Left wrist
                    left_wrist_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                             keyps=keyps, joint_name='left_wrist', fr_no=i, pid=j)
                    left_wrist_im = colorize(left_wrist_im, total_fr, fr_no=i, joint=1)
                    left_wrist_mo = left_wrist_mo + left_wrist_im[0]

                    ##### Right wrist
                    right_wrist_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                              keyps=keyps, joint_name='right_wrist', fr_no=i, pid=j)
                    right_wrist_im = colorize(right_wrist_im, total_fr, fr_no=i, joint=1)
                    right_wrist_mo = right_wrist_mo + right_wrist_im[0]

                    ###### Left Hip
                    left_hip_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                           keyps=keyps, joint_name='left_hip', fr_no=i, pid=j)
                    left_hip_im = colorize(left_hip_im, total_fr, fr_no=i, joint=2)
                    left_hip_mo = left_hip_mo + left_hip_im[0]

                    ###### right Hip
                    right_hip_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                            keyps=keyps, joint_name='right_hip', fr_no=i, pid=j)
                    right_hip_im = colorize(right_hip_im, total_fr, fr_no=i, joint=2)
                    right_hip_mo = right_hip_mo + right_hip_im[0]

                    ###### Left knee
                    left_knee_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                            keyps=keyps, joint_name='left_knee', fr_no=i, pid=j)
                    left_knee_im = colorize(left_knee_im, total_fr, fr_no=i, joint=2)
                    left_knee_mo = left_knee_mo + left_knee_im[0]

                    ###### Right knee
                    right_knee_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                             keyps=keyps, joint_name='right_knee', fr_no=i, pid=j)
                    right_knee_im = colorize(right_knee_im, total_fr, fr_no=i, joint=2)
                    right_knee_mo = right_knee_mo + right_knee_im[0]

                    ###### Left ankle
                    left_ankle_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                             keyps=keyps, joint_name='left_ankle', fr_no=i, pid=j)
                    left_ankle_im = colorize(left_ankle_im, total_fr, fr_no=i, joint=2)
                    left_ankle_mo = left_ankle_mo + left_ankle_im[0]

                    ##### Right ankle
                    right_ankle_im = joint_im(im_size=(ch, img_size[0], img_size[1]), ksize=ksize, kernel=gauss_ker_2d,
                                              keyps=keyps, joint_name='right_ankle', fr_no=i, pid=j)
                    right_ankle_im = colorize(right_ankle_im, total_fr, fr_no=i, joint=2)
                    right_ankle_mo = right_ankle_mo + right_ankle_im[0]
    
    nose_mo = nose_mo / (np.amax(nose_mo) + eps)
    neck_mo = neck_mo / (np.amax(neck_mo) + eps)
    top_mo = top_mo / (np.amax(top_mo) + eps)
    left_shoulder_mo = left_shoulder_mo / (np.amax(left_shoulder_mo) + eps)
    right_shoulder_mo = right_shoulder_mo / (np.amax(right_shoulder_mo) + eps)
    left_elbow_mo = left_elbow_mo / (np.amax(left_elbow_mo) + eps)
    right_elbow_mo = right_elbow_mo / (np.amax(right_elbow_mo) + eps)
    left_wrist_mo = left_wrist_mo / (np.amax(left_wrist_mo) + eps)
    right_wrist_mo = right_wrist_mo / (np.amax(right_wrist_mo) + eps)
    left_hip_mo = left_hip_mo / (np.amax(left_hip_mo) + eps)
    right_hip_mo = right_hip_mo / (np.amax(right_hip_mo) + eps)
    left_knee_mo = left_knee_mo / (np.amax(left_knee_mo) + eps)
    right_knee_mo = right_knee_mo / (np.amax(right_knee_mo) + eps)
    left_ankle_mo = left_ankle_mo / (np.amax(left_ankle_mo) + eps)
    right_ankle_mo = right_ankle_mo / (np.amax(right_ankle_mo) + eps)
    parts = [nose_mo, neck_mo, top_mo, left_shoulder_mo, right_shoulder_mo, left_elbow_mo, right_elbow_mo,
                    left_wrist_mo, right_wrist_mo, left_hip_mo, right_hip_mo, left_knee_mo, right_knee_mo,
                    left_ankle_mo, right_ankle_mo]
    pose_mo = np.zeros((3, height, width))
    for i in range(len(parts)):
        pose_mo[0] += parts[i][0]
        pose_mo[1] += parts[i][1]
        pose_mo[2] += parts[i][2]
        
    pose_mo = pose_mo.transpose(2,1,0)*255
    
    if np.sum(pose_mo) <= eps:
        return None
    else:
        pose_mo = np.ones((pose_mo.shape)) + np.random.rand(height, width, ch) + pose_mo
        pose_mo[pose_mo>255] = 255
        return pose_mo


def write_data(labels_file_path, data_path):
    count = 0
    labels_handler = open(labels_file_path,'r')
    for i,line in enumerate(labels_handler.readlines()):
        line = line.rstrip()
        clip_path = '/media'+line.split()[0]
        print(clip_path)
        if os.path.exists(clip_path):
            original_size = eval(line.split()[1]+line.split()[2])
            clip_label = eval(line.split()[-1])
            potion_data = None
            potion_data = PoTion(clip_path, original_size=original_size, img_size=(height, width), ch=col_ch, sigma=sigma, scale=resize_scale)
            """
            try:
                potion_data = PoTion(clip_path, original_size=original_size, img_size=(height, width), ch=col_ch, sigma=sigma, scale=resize_scale)
            except:
                print 'Failed'
            """
            if potion_data is not None:
                count+=1
                print(count)
                if clip_label==0:
                    img_file = data_path + 'Typical/' + str(count) + '.jpg'
                else:
                    img_file = data_path + 'Atypical/' + str(count) + '.jpg'
                cv.imwrite(img_file, potion_data)
                
    labels_handler.close()


training_labels_path = 'train_label.txt'
testing_labels_path = 'test_label.txt'
training_data_path = 'potion_data_pf/train/'
testing_data_path = 'potion_data_pf/test/'

write_data(training_labels_path, training_data_path)
write_data(testing_labels_path, testing_data_path)
