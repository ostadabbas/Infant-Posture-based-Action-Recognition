import json
import matplotlib
import numpy as np
#import matplotlib.pyplot as plt
import os
import argparse
#import matplotlib.patches as patches
#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
#from matplotlib import rc
#from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random


#rc('text', usetex = True)
#rc('text.latex', preamble = '\usepackage{color}')

posture_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/ori_posture_model_outputs/posture_2d_res'
vid_list = os.listdir(posture_root)
print(vid_list)

posture3d_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/ori_posture_model_outputs/posture_3d_res'

tar_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset_new'

anno_file = '/work/aclab/xiaof.huang/InfantActionData/InfAct_anno.csv'

data = pd.read_csv(anno_file)
anno_df = pd.DataFrame(data, columns=['Posture 1 Class','Posture 1 Start Frame','Posture 1 End Frame','Posture 3 Class','Posture 3 Start Frame','Posture 3 End Frame'])
anno = anno_df.to_numpy()

select_frames = np.zeros((400, 2))
select_preds_2d = []
select_preds_3d = []
select_labels = []


action_map = {'1':[0,1],'2':[0,2],'3':[0,3],'4':[0,4],'5':[1,0],'6':[1,2],'7':[1,3],'8':[1,4],'9':[2,0],'10':[2,1],'11':[2,3],'12':[2,4],'13':[3,0],'14':[3,1],'15':[3,2],'16':[3,4],'17':[4,0],'18':[4,1],'19':[4,2],'20':[4,3]}
keylist = list(action_map.keys())
valuelist = list(action_map.values())

action_200_arr = np.zeros((200, 14))

#action_160_classes1 = [1, 5, 9, 11, 10, 12, 15, 14, 20, 19]
#action_160_classes2 = [1, 5, 9, 11, 10, 12, 15, 16, 20, 19]
selected_action_classes = [1, 5, 8, 9, 10, 11, 15, 16, 19]

filteredout_videos = [3, 43, 46, 53, 138, 197]

i = 0
for vid in range(len(vid_list)):
    vid_name = vid_list[vid]
    if int(vid_name) in filteredout_videos:
        continue
    pred1_folder = os.path.join(posture_root, vid_name)

    tar1 = np.load(os.path.join(pred1_folder, 'tar.npy'),allow_pickle=True)
    imgs1 = np.load(os.path.join(pred1_folder, 'img.npy'),allow_pickle=True)
    pred1 = np.load(os.path.join(pred1_folder, 'pred.npy'),allow_pickle=True)    
    scores1 = np.load(os.path.join(pred1_folder, 'score.npy'),allow_pickle=True)
    imgs1 = [element for sublist in imgs1 for element in sublist]
    pred1 = [element.cpu().numpy() for sublist in pred1 for element in sublist]
    pred1 = np.array(pred1).flatten()
    score1 = scores1[0]
    for ele in range(1, len(scores1)):
        score1 = np.concatenate((score1, scores1[ele]))

    pred2_folder = os.path.join(posture3d_root, vid_name)

    tar2 = np.load(os.path.join(pred2_folder, 'tar.npy'),allow_pickle=True)
    imgs2 = np.load(os.path.join(pred2_folder, 'img.npy'),allow_pickle=True)
    pred2 = np.load(os.path.join(pred2_folder, 'pred.npy'),allow_pickle=True)    
    scores2 = np.load(os.path.join(pred2_folder, 'score.npy'),allow_pickle=True)
    imgs2 = [element for sublist in imgs2 for element in sublist]
    pred2 = [element.cpu().numpy() for sublist in pred2 for element in sublist]
    pred2 = np.array(pred2).flatten()  
    score2 = scores2[0]
    for ele in range(1, len(scores2)):
        score2 = np.concatenate((score2, scores2[ele]))

    onset_idx = anno[int(vid_name)-1][2]
    offset_idx = anno[int(vid_name)-1][4]
    start_posture = str(anno[int(vid_name)-1][0])   
    end_posture = str(anno[int(vid_name)-1][3])

    select_vid = int(vid_name)
    frame1 = int(onset_idx/2)
    frame2 = int((len(imgs1) - offset_idx)/2) + offset_idx
    select_frames[i*2][0] = select_vid
    select_frames[i*2][1] = frame1
    select_frames[i*2+1][0] = select_vid
    select_frames[i*2+1][1] = frame2

    select_preds_2d.append(pred1[frame1])
    select_preds_2d.append(pred1[frame2])
    pred1_postures = [pred1[frame1],pred1[frame2]]

    select_preds_3d.append(pred2[frame1])
    select_preds_3d.append(pred2[frame2])
    pred2_postures = [pred2[frame1],pred2[frame2]]


    select_labels.append(int(start_posture))
    select_labels.append(int(end_posture))
    gt_postures = [int(start_posture),int(end_posture)]

    action_200_arr[i][0] = int(vid_name)
    action_200_arr[i][4] = onset_idx
    action_200_arr[i][5] = offset_idx - onset_idx
    action_200_arr[i][6] = len(imgs1) - offset_idx
    action_200_arr[i][7] = len(imgs1)

    action_200_arr[i][8] = int(start_posture)
    action_200_arr[i][9] = int(end_posture)

    action_200_arr[i][10] = int(pred1[frame1])
    action_200_arr[i][11] = int(pred1[frame2])

    action_200_arr[i][12] = int(pred2[frame1])
    action_200_arr[i][13] = int(pred2[frame2])

    
    for j in range(len(keylist)):
        if valuelist[j] == gt_postures:
            action_200_arr[i][1] = int(keylist[j])
        
        if valuelist[j] == pred1_postures:
            action_200_arr[i][2] = int(keylist[j])

        if valuelist[j] == pred2_postures:
            action_200_arr[i][3] = int(keylist[j])
    i = i + 1
print(action_200_arr)
np.savetxt(os.path.join(tar_root, 'action_200_arr.txt'), action_200_arr, delimiter=',', fmt='%d')

print(np.unique(action_200_arr[:,1], return_counts=True))


#idx_160_arr1 = np.isin(action_200_arr[:,1], np.array(action_160_classes2))
idx_156_arr1 = np.isin(action_200_arr[:,1], np.array(selected_action_classes))

action_156_arr = action_200_arr[idx_156_arr1]
#print(action_156_arr[:,1])
#print(action_156_arr[:,2])
mask_2d = np.logical_and(action_156_arr[:,1] == action_156_arr[:,2], action_156_arr[:,1] == 1)
mask_3d = np.logical_and(action_156_arr[:,1] == action_156_arr[:,3], action_156_arr[:,1] == 1)
action_2d_acc = np.sum(mask_2d)
action_3d_acc = np.sum(mask_3d)
print(action_2d_acc)
print(action_3d_acc)

action_50_1_arr = action_156_arr[action_156_arr[:,1] == 1]
arr_1 = action_50_1_arr[random.sample(range(0,action_50_1_arr.shape[0]-1), 7)]
print(arr_1.shape)

action_50_5_arr = action_156_arr[action_156_arr[:,1] == 5]
arr_2 = action_50_5_arr[random.sample(range(0,action_50_5_arr.shape[0]-1), 5)]
print(arr_2.shape)

action_50_8_arr = action_156_arr[action_156_arr[:,1] == 8]
arr_3 = action_50_8_arr[random.sample(range(0,action_50_8_arr.shape[0]-1), 4)]
print(arr_3.shape)

action_50_9_arr = action_156_arr[action_156_arr[:,1] == 9]
arr_4 = action_50_9_arr[random.sample(range(0,action_50_9_arr.shape[0]-1), 4)]
print(arr_4.shape)

action_50_10_arr = action_156_arr[action_156_arr[:,1] == 10]
arr_5 = action_50_10_arr[random.sample(range(0,action_50_10_arr.shape[0]-1), 4)]
print(arr_5.shape)

action_50_11_arr = action_156_arr[action_156_arr[:,1] == 11]
arr_6 = action_50_11_arr[random.sample(range(0,action_50_11_arr.shape[0]-1), 8)]
print(arr_6.shape)

action_50_15_arr = action_156_arr[action_156_arr[:,1] == 15]
arr_7 = action_50_15_arr[random.sample(range(0,action_50_15_arr.shape[0]-1), 7)]
print(arr_7.shape)

action_50_16_arr = action_156_arr[action_156_arr[:,1] == 16]
arr_8 = action_50_16_arr[random.sample(range(0,action_50_16_arr.shape[0]-1), 3)]
print(arr_8.shape)

action_50_19_arr = action_156_arr[action_156_arr[:,1] == 19]
arr_9 = action_50_19_arr[random.sample(range(0,action_50_19_arr.shape[0]-1), 8)]
print(arr_9.shape)

arr_tmp = np.concatenate((arr_1, arr_2), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_3), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_4), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_5), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_6), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_7), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_8), axis=0)
arr = np.concatenate((arr_tmp, arr_9), axis=0)
print(arr.shape)


'''
action_60_1_arr = action_160_arr[action_160_arr[:,1] == 1]
tmp_arr1 = action_60_1_arr[action_60_1_arr[:,1] == action_60_1_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 6)]
tmp_arr2 = action_60_1_arr[action_60_1_arr[:,1] != action_60_1_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 1)]
arr_1 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_1.shape)

action_60_5_arr = action_160_arr[action_160_arr[:,1] == 5]
tmp_arr1 = action_60_5_arr[action_60_5_arr[:,1] == action_60_5_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 5)]
tmp_arr2 = action_60_5_arr[action_60_5_arr[:,1] != action_60_5_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 0)]
arr_5 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_5.shape)


action_60_9_arr = action_160_arr[action_160_arr[:,1] == 9]
tmp_arr1 = action_60_9_arr[action_60_9_arr[:,1] == action_60_9_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 6)]
tmp_arr2 = action_60_9_arr[action_60_9_arr[:,1] != action_60_9_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 0)]
arr_9 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_9.shape)


action_60_11_arr = action_160_arr[action_160_arr[:,1] == 11]
tmp_arr1 = action_60_11_arr[action_60_11_arr[:,1] == action_60_11_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 4)]
tmp_arr2 = action_60_11_arr[action_60_11_arr[:,1] != action_60_11_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 4)]
arr_11 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_11.shape)


action_60_10_arr = action_160_arr[action_160_arr[:,1] == 10]
tmp_arr1 = action_60_10_arr[action_60_10_arr[:,1] == action_60_10_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 3)]
tmp_arr2 = action_60_10_arr[action_60_10_arr[:,1] != action_60_10_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 2)]
arr_10 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_10.shape)


action_60_12_arr = action_160_arr[action_160_arr[:,1] == 12]
tmp_arr1 = action_60_12_arr[action_60_12_arr[:,1] == action_60_12_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 3)]
tmp_arr2 = action_60_12_arr[action_60_12_arr[:,1] != action_60_12_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 1)]
arr_12 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_12.shape)


action_60_15_arr = action_160_arr[action_160_arr[:,1] == 15]
tmp_arr1 = action_60_15_arr[action_60_15_arr[:,1] == action_60_15_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 4)]
tmp_arr2 = action_60_15_arr[action_60_15_arr[:,1] != action_60_15_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 5)]
arr_15 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_15.shape)

action_60_16_arr = action_160_arr[action_160_arr[:,1] == 16]
tmp_arr1 = action_60_16_arr[action_60_16_arr[:,1] == action_60_16_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 3)]
tmp_arr2 = action_60_16_arr[action_60_16_arr[:,1] != action_60_16_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 0)]
arr_16 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_16.shape)


action_60_20_arr = action_160_arr[action_160_arr[:,1] == 20]
tmp_arr1 = action_60_20_arr[action_60_20_arr[:,1] == action_60_20_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 3)]
tmp_arr2 = action_60_20_arr[action_60_20_arr[:,1] != action_60_20_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 2)]
arr_20 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_20.shape)


action_60_19_arr = action_160_arr[action_160_arr[:,1] == 19]
tmp_arr1 = action_60_19_arr[action_60_19_arr[:,1] == action_60_19_arr[:,3]]
tmp_arr1 = tmp_arr1[random.sample(range(0,tmp_arr1.shape[0]-1), 6)]
tmp_arr2 = action_60_19_arr[action_60_19_arr[:,1] != action_60_19_arr[:,3]]
tmp_arr2 = tmp_arr2[random.sample(range(0,tmp_arr2.shape[0]-1), 2)]
arr_19 = np.concatenate((tmp_arr1, tmp_arr2), axis=0)
print(arr_19.shape)

arr_tmp = np.concatenate((arr_1, arr_5), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_9), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_11), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_10), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_12), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_15), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_16), axis=0)
arr_tmp = np.concatenate((arr_tmp, arr_20), axis=0)
arr = np.concatenate((arr_tmp, arr_19), axis=0)
print(arr.shape)
'''

idx_list = np.isin(action_200_arr[:,0], action_156_arr[:,0], invert=True)
rest_arr = action_200_arr[idx_list]
print(rest_arr.shape)

idx_list = np.isin(action_156_arr[:,0], arr[:,0], invert=True)
train_arr = action_156_arr[idx_list]
print(train_arr.shape)
print(np.sum(train_arr[:,1] == train_arr[:,2])/106)
print(np.sum(train_arr[:,1] == train_arr[:,3])/106)

test_arr = arr
print(test_arr.shape)
print(np.sum(test_arr[:,1] == test_arr[:,2])/50)
print(np.sum(test_arr[:,1] == test_arr[:,3])/50)

with open(os.path.join(tar_root, 'rest_videos.npy'), 'wb') as f:
    np.save(f, rest_arr)

with open(os.path.join(tar_root, 'train_videos.npy'), 'wb') as f:
    np.save(f, train_arr)

with open(os.path.join(tar_root, 'test_videos.npy'), 'wb') as f:
    np.save(f, test_arr)


#postures_cm
mask = np.isin(test_arr[:,1], np.array(selected_action_classes))

select_preds_2d = list(test_arr[:,10:12].flatten())
select_preds_3d = list(test_arr[:,12:14].flatten())
select_labels = list(test_arr[:,8:10].flatten())
acc_2d = sum(1 for x,y in zip(select_preds_2d, select_labels) if x == y) / len(select_labels)
print(acc_2d)
acc_3d = sum(1 for x,y in zip(select_preds_3d, select_labels) if x == y) / len(select_labels)
print(acc_3d)


data_2d = {'y_actual': select_labels, 'y_predicted': select_preds_2d}
df_2d = pd.DataFrame(data_2d)

confusion_matrix_2d = pd.crosstab(df_2d['y_actual'], df_2d['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix_2d)

data_3d = {'y_actual': select_labels, 'y_predicted': select_preds_3d}
df_3d = pd.DataFrame(data_3d)

confusion_matrix_3d = pd.crosstab(df_3d['y_actual'], df_3d['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix_3d)

plt.figure()
plt.title('Predicted 2D Pose-based Posture')
sn.heatmap(confusion_matrix_2d, annot=True)
new_labels = ['Supine','Prone','Sitting','Standing','All-fours']
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=new_labels)
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=new_labels)

plt.savefig(os.path.join(tar_root, 'cm_posture_2d_test_video_frames.png'))
plt.close()

plt.figure()
plt.title('Predicted 3D Pose-based Posture')
sn.heatmap(confusion_matrix_3d, annot=True)
new_labels = ['Supine','Prone','Sitting','Standing','All-fours']
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=new_labels)
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], labels=new_labels)
  
plt.savefig(os.path.join(tar_root, 'cm_posture_3d_test_video_frames.png'))
plt.close()