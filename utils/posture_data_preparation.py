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

#rc('text', usetex = True)
#rc('text.latex', preamble = '\usepackage{color}')


posture_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/posture_2d_res'
vid_list = os.listdir(posture_root)
print(vid_list)

posture3d_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/posture_3d_res'

tar_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/posture_sig_vis'

anno_file = '/work/aclab/xiaof.huang/InfantActionData/209_labelwith180new.csv'

split_file = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos7_200_outputs/Infant_PoseC3d_segmented_postures_Y6.pkl'
with open(split_file, 'rb') as f:
    split_idx_list = pickle.load(f)

train_vids = set(Extract(split_idx_list['indices']['X_train']))
val_vids = set(Extract(split_idx_list['indices']['X_val']))
test_vids = set(Extract(split_idx_list['indices']['X_test']))
print(len(train_vids))
print(len(val_vids))
print(len(test_vids))

data = pd.read_csv(anno_file)
anno_df = pd.DataFrame(data, columns=['Posture 1 Class','Posture 1 Start Frame','Posture 1 End Frame','Posture 3 Class','Posture 3 Start Frame','Posture 3 End Frame'])
anno = anno_df.to_numpy()

Posture_processed_data = {'train_data':{'X':[], 'vid_name':[], 'vid_len':[], }, 'val_data':{}, 'test_data':{}}


train_vids_len = []
val_vids_len = []
test_vids_len = []

train_X = []
val_X = []
test_X = []

for i in range(len(vid_list)):
    vid_name = vid_list[i]
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



    t = np.arange(0, len(imgs1))

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(t, score1[:, 0], c='r', label='Supine')
    ax1.plot(t, score1[:, 1], c='b', label='Prone')
    ax1.plot(t, score1[:, 2], c='g', label='Sitting')
    ax1.plot(t, score1[:, 3], c='c', label='Standing')
    ax1.plot(t, score1[:, 4], c='m', label='All Fours')
    ax1.axvline(x = anno[int(vid_name)-1][2], color = 'k', linewidth=2, label = 'gt_start_'+str(anno[int(vid_name)-1][0]))
    ax1.axvline(x = anno[int(vid_name)-1][4], color = 'k', linewidth=2, label = 'gt_end_'+str(anno[int(vid_name)-1][3]))
    ax1.set_title('Posture Signal Inferred by 2D Pose-Based Posture Classifier')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Confidence Score')

    ax2.plot(t, score2[:, 0], c='r', label='Supine')
    ax2.plot(t, score2[:, 1], c='b', label='Prone')
    ax2.plot(t, score2[:, 2], c='g', label='Sitting')
    ax2.plot(t, score2[:, 3], c='c', label='Standing')
    ax2.plot(t, score2[:, 4], c='m', label='All Fours')
    ax2.axvline(x = anno[int(vid_name)-1][2], color = 'k', linewidth=2, label = 'gt_start_'+str(anno[int(vid_name)-1][0]))
    ax2.axvline(x = anno[int(vid_name)-1][4], color = 'k', linewidth=2, label = 'gt_end_'+str(anno[int(vid_name)-1][3]))
    ax2.set_title('Posture Signal Inferred by 3D Pose-Based Posture Classifier')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Confidence Score')
    
    plt.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(tar_root, 'vid_'+str(vid_name)+'_posture_signal.png'))
    
    #plt.show()
    plt.close()

