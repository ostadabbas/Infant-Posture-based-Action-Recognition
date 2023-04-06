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

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

posture_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs/posture_2d_res'
vid_list = os.listdir(posture_root)
print(vid_list)

posture3d_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs/posture_3d_res'

tar_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs/posture_sig_vis'

anno_file = '/work/aclab/xiaof.huang/InfantActionData/InfAct_anno.csv'

data = pd.read_csv(anno_file)
anno_df = pd.DataFrame(data, columns=['Posture 1 Class','Posture 1 Start Frame','Posture 1 End Frame','Posture 3 Class','Posture 3 Start Frame','Posture 3 End Frame'])
anno = anno_df.to_numpy()

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
    #ax1.axvline(x = 141, color = 'k', linewidth=2, label = 'gt_start_'+str(anno[int(vid_name)-1][0]))
    #ax1.axvline(x = 191, color = 'k', linewidth=2, label = 'gt_end_'+str(anno[int(vid_name)-1][3]))

    ax1.set_title('Posture Signal Inferred by 2D Pose-Based Posture Classifier')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Confidence Score')
    set_size(6,2.5)

    ax2.plot(t, score2[:, 0], c='r', label='Supine')
    ax2.plot(t, score2[:, 1], c='b', label='Prone')
    ax2.plot(t, score2[:, 2], c='g', label='Sitting')
    ax2.plot(t, score2[:, 3], c='c', label='Standing')
    ax2.plot(t, score2[:, 4], c='m', label='All Fours')
    ax2.axvline(x = anno[int(vid_name)-1][2], color = 'k', linewidth=2, label = 'gt_start_'+str(anno[int(vid_name)-1][0]))
    ax2.axvline(x = anno[int(vid_name)-1][4], color = 'k', linewidth=2, label = 'gt_end_'+str(anno[int(vid_name)-1][3]))
    #ax2.axvline(x = 141, color = 'k', linewidth=2, label = 'gt_start_'+str(anno[int(vid_name)-1][0]))
    #ax2.axvline(x = 191, color = 'k', linewidth=2, label = 'gt_end_'+str(anno[int(vid_name)-1][3]))

    ax2.set_title('Posture Signal Inferred by 3D Pose-Based Posture Classifier')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Confidence Score')
    set_size(6,2.5)

    #plt.legend()
    # Put a legend below current axis
    #ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=6)
    fig.tight_layout()
    plt.savefig(os.path.join(tar_root, 'vid_'+str(vid_name)+'_posture_signal.png'))
    
    #plt.show()
    plt.close()

