import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.patches as patches
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from matplotlib import rc
from matplotlib.patches import Rectangle

#rc('text', usetex = True)
#rc('text.latex', preamble = '\usepackage{color}')

def show_posture_img(img, img_name, output_path, posture_p, score_p):
    h = img.shape[0]
    w = img.shape[1]
    print(w, h)
    plt.imshow(img)

    posture_label = ['Supine', 'Prone', 'Sitting', 'Standing']

    idx_list = np.argsort(score_p)
    str1 = posture_label[idx_list[3]] + ':%.4f'% score_p[idx_list[3]]
    str2 = posture_label[idx_list[2]] + ':%.4f'% score_p[idx_list[2]]
    str3 = posture_label[idx_list[1]] + ':%.4f'% score_p[idx_list[1]]
    str4 = posture_label[idx_list[0]] + ':%.4f'% score_p[idx_list[0]]

    posture_anno = False
    if posture_anno == True:
        if posture_t == posture_p:
            c = 'g'
        else:
            c = 'r'
    else:
        c = 'k'
    plt.gca().add_patch(Rectangle((0, h-(4*40+10)), 300, 4*40+10,  alpha = 0.4, facecolor='w', edgecolor='none'))
   
    plt.text(0, h-40*3, str1, ha = 'left', va = 'bottom', size = 10, fontweight = 'bold', color = c)
    plt.text(0, h-40*2, str2, ha = 'left',  va = 'bottom', size = 10, color = 'k')
    plt.text(0, h-40*1, str3, ha = 'left',  va = 'bottom', size = 10, color = 'k')
    plt.text(0, h-40*0, str4, ha = 'left',  va = 'bottom', size = 10, color = 'k')

    plt.axis('off')

    plt.savefig(os.path.join(output_path, img_name),bbox_inches = 'tight',dpi = 300, pad_inches=0.0)
    plt.close()


if __name__ == '__main__':
    pred_2d_postures_root = '/home/faye/Downloads/posture_2d_4class_res'
    pred_3d_postures_root = '/home/faye/Downloads/posture_3d_res'
    imgs_root = '/home/faye/Downloads/pose_2d_vis'
    output_root = '/home/faye/Downloads/posture_2d_4class_vis'
    vid_list = os.listdir(pred_2d_postures_root)
    print(vid_list)

    for i in range(2,len(vid_list)):
        vid_name = vid_list[i]
        pred_folder = os.path.join(pred_2d_postures_root, vid_name)

        img_path = os.path.join(imgs_root, vid_name)

        output_path = os.path.join(output_root, vid_name)    
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        tar = np.load(os.path.join(pred_folder, 'tar.npy'),allow_pickle=True)
        imgs = np.load(os.path.join(pred_folder, 'img.npy'),allow_pickle=True)
        pred = np.load(os.path.join(pred_folder, 'pred.npy'),allow_pickle=True)    
        scores = np.load(os.path.join(pred_folder, 'score.npy'),allow_pickle=True)
        imgs = [element for sublist in imgs for element in sublist]
        pred = [element.cpu().numpy() for sublist in pred for element in sublist]
        pred = np.array(pred).flatten()
        score = scores[0]
        for ele in range(1, len(scores)):
            score = np.concatenate((score, scores[ele]))

        for index in range(len(imgs)):
            print(index)
            img_name = 'test'+str(imgs[index])+'.jpg'
            print(img_name)
            posture_p = pred[index]
            score_p = score[index]

            img = plt.imread(os.path.join(img_path,img_name))
            show_posture_img(img, img_name, output_path, posture_p, score_p)

