import sys
import math
import copy
import numpy
import os
import pickle
import json
import os.path as osp
import numpy as np
import cv2 as cv
import glob
from ParticleFilter import particle_filter

DTYPE = np.float32

def PF(kps, h, w):
    keyps = np.zeros((len(kps), 4, 17))
    for k in range(len(kps)):
        if kps[k]!=[]:
            keyps[k] = kps[k][0]     
    ave = sum(keyps)[:2]/len(keyps)
    for j in [5,6,7,8,9,10,11,12,13,14,15,16]:
        initial_x = list(ave[:,j])
        initial_x.append(0)
        keyps = particle_filter(keyps, j, initial_x, h, w)
    
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

def json2pickle(data):
    keyps = np.zeros((len(data), 1, 4, 17))
  
    for i in range(len(data)):
        kpts = data[i]['keypoints']
        for j in range(17):
            if kpts[j*3+2] >= 0.8:
                 keyps[i][0][0][j] = kpts[j*3+0]
                 keyps[i][0][1][j] = kpts[j*3+1]
    return keyps   

def pickle2json(ori_data, keyps):
    for i in range(len(keyps)):
        for j in range(17):
            ori_data[i]['keypoints'][j*3+0] = keyps[i][0][0][j] 
            ori_data[i]['keypoints'][j*3+1] = keyps[i][0][1][j]
    return ori_data      

def main(pred_file, tar_file, height, width):
    with open(pred_file) as f:
        data = json.load(f)

    #convert prediction json format to pickle structure 
    keyps = json2pickle(data)

    total_fr = len(keyps)
    if total_fr<60:
        return None
    
    n_missing = 0
    for i in range(total_fr):
        if np.sum(keyps[i])==0:
            n_missing+=1
            
    if n_missing>total_fr*0.9:
        return None
        
    '''    
    original_h, original_w = original_size
    for i,k1 in enumerate(keyps):
        for j,k2 in enumerate(k1):
            keyps[i][j][0] = k2[0] / original_h * height
            keyps[i][j][1] = k2[1] / original_w * width
    '''
    keyps = Interpolation(keyps)
    keyps=PF(keyps, h, w)  
    '''
    if n_missing!=0:
        print("Interpolation")
        keyps = Interpolation(keyps)
        
    if n_missing>total_fr*0.1:
        print("ParticleFilter")
        keyps=PF(keyps, h, w)
    '''
    #convert pickle structure to json format
    final_data = pickle2json(data, keyps)

    with open(tar_file, 'w') as fp:
        json.dump(final_data, fp)


pred_file = '/home/faye/Documents/InfantProject/outputs/example3_outputs/yolov3_example3_results.json'
tar_file = '/home/faye/Documents/InfantProject/outputs/example3_outputs/yolov3_example3_results_PF.json'


pred_file = "/home/faye/Documents/InfantProject/outputs/example3_outputs/keypoints_validate_infant_results_0.json"
tar_file = "/home/faye/Documents/InfantProject/outputs/example3_outputs/keypoints_validate_infant_results_video_IP.json"
h = 279
w = 496
main(pred_file, tar_file, h , w)
     
   
