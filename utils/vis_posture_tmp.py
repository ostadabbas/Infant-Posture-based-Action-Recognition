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


"""

python vis_posture_tmp.py --data_info_home /home/faye/Documents/InfantProject/outputs/SyRIP100_outputs --pred_folder /home/faye/Documents/FiDIP_Posture/Classifier/test700_withoutTrans_test_kpts_output_202207281336 --img_folder /home/faye/Documents/human_body_prior/symmetry_measurement/SyRIP_data/images/new_SyRIP_700 --output_folder posture_vis_withTrans --has_anno False


"""
def show_2d_img(img,kpts,edges,color,file_name,output, posture_p, posture_t, score_p, posture_anno=False):
    num_joints = kpts.shape[0]
    num_edge = len(edges)
    h = img.shape[0]
    w = img.shape[1]
    print(w, h)
    plt.imshow(img)
    
    '''
    currentAxis = plt.gca()
    rect = patches.Rectangle([bbox[0],bbox[1]], bbox[2], bbox[3], linewidth=1,edgecolor='b',facecolor='none')
    currentAxis.add_patch(rect)
    ''' 
    '''    
    for i in range(num_edge):
        x_list = []
        y_list = []

        # if kpts[edges[i,0],2] == 2 and kpts[edges[i,1],2] == 2:
        x_list.append(kpts[edges[i,0],0])
        x_list.append(kpts[edges[i,1],0])

        y_list.append(kpts[edges[i,0],1])
        y_list.append(kpts[edges[i,1],1])
       
        plt.plot(x_list, y_list,color = color[i], linewidth = 2)


    for a in range(num_joints):
        if a == 0:
            continue
        if a == 1:
            continue
        if a == 2:
            continue
        if a == 3:
            continue
        if a == 4:
            continue
        else:
        # if kpts[a,2] == 2:
            plt.plot(kpts[a,0], kpts[a,1],'r.') 
    '''

    '''    
    textstr = '\n'.join((
              r'$Supine: %.2f$' % score_p[0],
              r'$Prone: %.2f$' % score_p[1],
              r'$Sitting: %.2f$' % score_p[2],
              r'$Standing: %.2f$' % score_p[3]))
    '''

    # LaTeX \newline doesn't work, but we can add multiple lines together
    # pgf_with_latex = {
    #     "text.usetex": True,
    #     "pgf.rcfonts": False,
    #     "pgf.preamble":[r'\usepackage{color}']}
    # matplotlib.rcParams.update(pgf_with_latex)
    '''
    annot1_txt = r'\textcolor{red}{Supine: %.5f}' % (score_p[0])
    annot1_txt += '\n'
    annot1_txt += r'\textcolor{yellow}{Prone: %.5f}' % (score_p[1])
    annot1_txt += '\n'
    annot1_txt += r'\textcolor{red}{Sitting: %.5f}' % (score_p[2])
    annot1_txt += '\n'
    annot1_txt += r'\textcolor{red}{Standing: %.5f}' % (score_p[3])
    '''
    '''
    color_test = ['yellow','yellow','yellow','yellow']
    if posture_t == posture_p:
        color_test[posture_p] = 'green'
        plt.text(30,900, annot1_txt.split('\n'), color_test, bbox = dict(facecolor = 'w',alpha = 0.8))
    '''

    posture_label = ['Supine', 'Prone', 'Sitting', 'Standing']


    idx_list = np.argsort(score_p)
    str1 = posture_label[idx_list[3]] + ':%.4f'% score_p[idx_list[3]]
    str2 = posture_label[idx_list[2]] + ':%.4f'% score_p[idx_list[2]]
    str3 = posture_label[idx_list[1]] + ':%.4f'% score_p[idx_list[1]]
    str4 = posture_label[idx_list[0]] + ':%.4f'% score_p[idx_list[0]]
    if posture_anno == True:
        if posture_t == posture_p:
            c = 'g'
        else:
            c = 'r'
    else:
        c = 'k'
    plt.gca().add_patch(Rectangle((0, 1100), 450, 200,  alpha = 0.3, facecolor='w', edgecolor='none'))
    #plt.text(30,900, annot1_txt, bbox = dict(facecolor = 'w',alpha = 0.8))
    plt.text(0, 1160, str1, ha = 'left', va = 'bottom', size = 10, fontweight = 'bold', color = c)
    plt.text(0, 1200, str2, ha = 'left',  va = 'bottom', size = 10, color = 'k')
    plt.text(0, 1240, str3, ha = 'left',  va = 'bottom', size = 10, color = 'k')
    plt.text(0, 1280, str4, ha = 'left',  va = 'bottom', size = 10, color = 'k')
    plt.axis('off')
    # plt.text(30,950, 'Supine-' + str(posture_p[0]), size = 20, color = 'r', bbox = dict(facecolor = 'r',alpha = 0.2))
    #plt.show()
    plt.savefig(os.path.join(output, file_name),bbox_inches = 'tight',dpi = 300, pad_inches=0.0)
    plt.close()

'''
edges = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7],
         [7, 9], [6, 8], [8, 10], [11, 12], [11, 13], [13, 15],
         [12, 14], [14,16]]  # 17 keypoints format
'''
edges = [[5, 6], [5, 7],
         [7, 9], [6, 8], [8, 10], [11, 12], [11, 13], [13, 15],
         [12, 14], [14,16]]  # 17 keypoints format
edges = np.array(edges)
cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, len(edges) + 5)]
colors = [np.array((c[2], c[1], c[0])) for c in colors]

if __name__ == '__main__':
    # A way to deal with the process output without needing to be root
    mask = 0o000
    umask = os.umask(mask)
    parser = argparse.ArgumentParser(description='Creating json file of test data for fidip')
    parser.add_argument('--data_info_home', metavar='path', required=True,
                        help='Root path of data')
    parser.add_argument('--anno_file', metavar='string', default='annotations/person_keypoints_validate_infant.json',
                        help='Should be the json file of annotation')
    parser.add_argument('--pred_folder', metavar='string', default='annotations/person_keypoints_validate_infant.json',
                        help='Should be the json file of annotation')
    parser.add_argument('--img_folder', metavar='string', default='images/validate_infant',
                        help='Images folder')
    parser.add_argument('--output_folder', metavar='string', required=True,
                        help='Output folder')
    parser.add_argument('--has_anno', metavar='boolean', required=True, default=False,
                        help='Output folder')
    args = parser.parse_args()
    anno_path = os.path.join(args.data_info_home, args.anno_file)
    img_path = os.path.join(args.data_info_home, args.img_folder)
    # anno_path = args.anno_file
    # img_path = args.img_folder
    output_path = os.path.join(args.data_info_home,args.output_folder)    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    tar = np.load(os.path.join(args.pred_folder, 'tar.npy'),allow_pickle=True)
    imgs = np.load(os.path.join(args.pred_folder, 'img.npy'),allow_pickle=True)
    pred = np.load(os.path.join(args.pred_folder, 'pred.npy'),allow_pickle=True)    
    scores = np.load(os.path.join(args.pred_folder, 'score.npy'),allow_pickle=True)
    tar = tar[0]
    pred = pred[0]
    imgs = imgs[0]
    scores = scores[0]

    for index in range(len(imgs)):
        print(index)
        img_name = imgs[index]
        print(img_name)
        posture_p = pred[index]
        posture_t = tar[index]
        score_p = scores[index]
        kpt = np.zeros((17,3))
        '''
        for i in range(len(kpt_info)):
            if img_name == 'test' + str(kpt_info[i]['image_id']) + '.jpg':
                tmps = kpt_info[i]['keypoints']
                kpt[:,0] = tmps[0::3]
                kpt[:,1] = tmps[1::3]
                kpt[:,2] = tmps[2::3]
                break
        '''
        # for i in range(len(kpt_info)):
        #     tmps = kpt_info[i]['keypoints']
        #     kpt[:,0] = tmps[0::3]
        #     kpt[:,1] = tmps[1::3]
        #     kpt[:,2] = tmps[2::3]
                

        img = plt.imread(os.path.join(img_path,img_name))
        show_2d_img(img,kpt,edges,colors,img_name,output_path, posture_p, posture_t, score_p, args.has_anno)

        
'''

    for pth, dir_list, file_list in IMG:
        for file_name in img:
            print('file_name: ', file_name)
            print('img_name: ', img_name[i])
            for i in range(len(kpts)):
                if file_name == 'test' + img_name[i] + '.jpg':
                    print('a')
                    index = int(img_name[i])
                    
                    if pred[index] == 0:
                        posture_p = 'Supine'
                    if pred[index] == 1:
                        posture_p = 'Prone'
                    if pred[index] == 2:
                        posture_p = 'Sitting'
                    if pred[index] == 3:
                        posture_p = 'Standing'
                    
                    posture_p = pred[index]
                    posture_t = tar[index]
                    score_p = scores[index]
                    img = plt.imread(os.path.join(img_path,file_name))
                    show_2d_img(img,kpts[i],edges,colors,file_name,output_path, posture_p, posture_t, score_p)
                    break
'''
