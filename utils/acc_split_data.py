import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.patches as patches
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from matplotlib import rc
from matplotlib.patches import Rectangle
import pandas as pd


def Extract(lst):
    return [item[0] for item in lst]

#rc('text', usetex = True)
#rc('text.latex', preamble = '\usepackage{color}')

dict = {'0':'Supine', '1':'Prone', '2':'Sitting', '3':'Standing', '4':'AllFours'}

split_file = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/Infant_PoseC3d_segmented_postures_Y6.pkl'
with open(split_file, 'rb') as f:
    split_idx_list = pickle.load(f)
#print(split_idx_list)
#print(len(split_idx_list['indices']['X_train']))
#print(len(split_idx_list['indices']['X_val']))
train_vids = set(Extract(split_idx_list['indices']['X_train']))
val_vids = set(Extract(split_idx_list['indices']['X_val']))
test_vids = set(Extract(split_idx_list['indices']['X_test']))
print(len(train_vids))
print(len(val_vids))
print(test_vids)

#res_file = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/209_labelwith180new.csv'
#res_data = pd.read_excel(res_file, index_col=0) 

anno_file = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/209_labelwith180new.csv'
data = pd.read_csv(anno_file)
anno_df = pd.DataFrame(data, columns=['Video Index','Posture 1 Class','Posture 2 Start Frame','Posture 2 End Frame','Posture 3 Class'])
anno = anno_df.to_numpy()
print(anno.shape)

tar_path = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/action_data_posture_prob_input/train'
gt = []
pred_2d = []
pred_3d = []
vid_name = []

# split data
vid_list = list(train_vids)
for i in range(len(vid_list)):
    idx_dict = vid_list[i]
    idx = int(idx_dict)-1

    # create X format
    subj_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/posture_2d_res/' + str(idx+1)
    tar = np.load(os.path.join(subj_fd, 'tar.npy'),allow_pickle=True)
    imgs = np.load(os.path.join(subj_fd, 'img.npy'),allow_pickle=True)
    pred = np.load(os.path.join(subj_fd, 'pred.npy'),allow_pickle=True)    
    scores = np.load(os.path.join(subj_fd, 'score.npy'),allow_pickle=True)
    imgs = [element for sublist in imgs for element in sublist]
    pred = [element.cpu().numpy() for sublist in pred for element in sublist]
    pred = np.array(pred).flatten()
    score = scores[0]
    for ele in range(1, len(scores)):
        score = np.concatenate((score, scores[ele]))
    print(imgs)
    print(score)

    col_posture = int(idx_dict[-1])
    if col_posture == 1:
        col_idx = 0
    elif col_posture == 3:
        col_idx = 1
    else:
        print('Error!')
    
    gt.append(anno[idx, col_idx])
    pred_2d.append(anno[idx, col_idx+2])
    pred_3d.append(anno[idx, col_idx+4])
    vid_name.append(idx_dict)
'''
# entire data
for i in range(anno.shape[0]):
    idx = i   
    gt.append(anno[idx, 0])
    pred_2d.append(anno[idx, 2])
    pred_3d.append(anno[idx, 4])
    vid_name.append('{0:03d}_1'.format(i))

    gt.append(anno[idx, 1])
    pred_2d.append(anno[idx, 3])
    pred_3d.append(anno[idx, 5])
    vid_name.append('{0:03d}_3'.format(i))

'''

print(len(gt))
array1 = np.array(gt)
array2 = np.array(pred_2d)
array3 = np.array(pred_3d)
subtracted_array1 = np.subtract(array1, array2)
subtracted1 = list(subtracted_array1)
subtracted_array2 = np.subtract(array1, array3)
subtracted2 = list(subtracted_array2)
print(subtracted1.count(0)/len(gt))
print(subtracted2.count(0)/len(gt))

anno_df1 = pd.DataFrame()
anno_df1['video_name'] = np.array(vid_name)
anno_df1['gt'] = np.array(gt)
anno_df1['pred_2d_xiaofei'] = np.array(pred_2d)
anno_df1['pred_3d_xiaofei'] = np.array(pred_3d)
anno_df1.to_csv('209_pred_comp_split_train.csv')
