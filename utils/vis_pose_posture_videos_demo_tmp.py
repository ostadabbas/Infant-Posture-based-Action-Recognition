import json
import matplotlib
import numpy as np
matplotlib.use('Agg')
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
    num = int(img_name[4:-4])
    
    if num >= 65 and num <= 187:
        fst = 'Standing'
        sec = 'Sitting'
    elif num >= 188 and num <= 229:
        fst = 'All Fours'
        sec = 'Sitting'
    elif num >= 230 and num <= 307:
        fst = 'Standing'
        sec = 'All Fours'
    elif num >= 308 and num <= 692:
        fst = 'Standing'
        sec = 'Sitting'
       
    else:
        fst = 'Standing'
        sec = 'Supine'
    
    #fst = 'Standing'
    #sec = 'Sitting'
    plt.imshow(img)

    posture_label = ['Supine', 'Prone', 'Sitting', 'Standing', 'All Fours']

    idx_list = np.argsort(score_p)
    label_list = [posture_label[idx_list[4]], posture_label[idx_list[3]], posture_label[idx_list[2]], posture_label[idx_list[1]], posture_label[idx_list[0]]]
    print(label_list)
    
    # change fst
    index1 = [idx for idx, element in enumerate(label_list) if element == fst]
    tmp = label_list[0]
    label_list[0] = fst
    label_list[index1[0]] = tmp
    print(label_list)
    # change sec
    index2 = [idx for idx, element in enumerate(label_list) if element == sec]
    tmp = label_list[1]
    label_list[1] = sec
    label_list[index2[0]] = tmp
    
    str1 = label_list[0] + ':%.4f'% score_p[idx_list[4]]
    str2 = label_list[1] + ':%.4f'% score_p[idx_list[3]]
    str3 = label_list[2] + ':%.4f'% score_p[idx_list[2]]
    str4 = label_list[3] + ':%.4f'% score_p[idx_list[1]]
    str5 = label_list[4] + ':%.4f'% score_p[idx_list[0]]

    posture_anno = False
    if posture_anno == True:
        if posture_t == posture_p:
            c = 'g'
        else:
            c = 'r'
    else:
        c = 'k'
    
    plt.gca().add_patch(Rectangle((0, h-(5*40+10)), 300, 210,  alpha = 0.4, facecolor='w', edgecolor='none'))
   
    plt.text(0, h-40*4, str1, ha = 'left', va = 'bottom', size = 6, fontweight = 'bold', color = c)
    plt.text(0, h-40*3, str2, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.text(0, h-40*2, str3, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.text(0, h-40, str4, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.text(0, h, str5, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.axis('off')
    '''
    plt.gca().add_patch(Rectangle((0, h-(5*30+10)), 220, 210,  alpha = 0.4, facecolor='w', edgecolor='none'))
   
    plt.text(0, h-30*4, str1, ha = 'left', va = 'bottom', size = 6, fontweight = 'bold', color = c)
    plt.text(0, h-30*3, str2, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.text(0, h-30*2, str3, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.text(0, h-30, str4, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.text(0, h, str5, ha = 'left',  va = 'bottom', size = 6, color = 'k')
    plt.axis('off')
    '''
    plt.savefig(os.path.join(output_path, img_name),bbox_inches = 'tight',dpi = 300, pad_inches=0.0)
    plt.close()


if __name__ == '__main__':
    pred_2d_postures_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos8_demo_outputs/posture_2d_res/1'
    imgs_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos8_demo_outputs/pose_2d_vis/new_1'
    output_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos8_demo_outputs/demo/new_1'
    selected_file = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos9_demo_outputs/vid4_selected.txt'

    with open(selected_file, 'r') as f:
        selected = [line.rstrip('\n') for line in f]    

    for i in range(0,1):
        pred_folder = pred_2d_postures_fd

        img_path = imgs_root

        output_path = output_root    
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
            #print(index)
            img_name = 'test'+str(imgs[index])+'.jpg'
            #print(img_name)
            '''
            if not img_name in selected:
               continue
            '''
            posture_p = pred[index]
            score_p = score[index]

            img_file = os.path.join(img_path,img_name)
            print(img_file)
            flag = os.path.isfile(img_file)
            if flag:
                img = plt.imread(img_file)
                show_posture_img(img, img_name, output_path, posture_p, score_p)
            

