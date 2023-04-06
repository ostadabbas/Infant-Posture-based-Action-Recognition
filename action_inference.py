import numpy as np
import pandas as pd
import os 
import torch
import seaborn as sn
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import preprocessing

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


def vis_signal(sq1, sq2, title, vid_name, tar_path):

    t1 = np.arange(0, len(sq1))

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(t1, sq1[:, 0], c='r', label='Supine')
    ax1.plot(t1, sq1[:, 1], c='b', label='Prone')
    ax1.plot(t1, sq1[:, 2], c='g', label='Sitting')
    ax1.plot(t1, sq1[:, 3], c='c', label='Standing')
    ax1.plot(t1, sq1[:, 4], c='m', label='All Fours')

    ax1.set_title('Start Signal of' + title)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Confidence Score')
    set_size(6,2.5)

    t2 = np.arange(0, len(sq2))

    ax2.plot(t2, sq2[:, 0], c='r', label='Supine')
    ax2.plot(t2, sq2[:, 1], c='b', label='Prone')
    ax2.plot(t2, sq2[:, 2], c='g', label='Sitting')
    ax2.plot(t2, sq2[:, 3], c='c', label='Standing')
    ax2.plot(t2, sq2[:, 4], c='m', label='All Fours')

    ax2.set_title('End Signal of' + title)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Confidence Score')
    set_size(6,2.5)

    #plt.legend()
    # Put a legend below current axis
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True, shadow=True, ncol=6)
    fig.tight_layout()
    plt.savefig(os.path.join(tar_path, 'vid_'+ vid_name + '_Signal ' + title + '.png'))
    
    #plt.show()
    plt.close()


def moving_average(v, n):
    moving_averages = []
    for m in range(n-1):
        moving_averages.append(v[m])
    i = 0
    while i < len(v) - n + 1:
        window = v[i : i + n]
        window_average = round(np.sum(window) / n, 2)
        moving_averages.append(window_average)	
        i += 1

    return moving_averages

def obj_weighted_moving_average(v, w, n):
    weighted_moving_averages = []
    for m in range(n-1):
        weighted_moving_averages.append(v[m])
    i = 0
    while i < len(v) - n + 1:
        window = v[i : i + n]
        #print(window)
        weights = [val/sum(w[i : i + n]) for val in w[i : i + n]]
        #print(weights)
        window_average = round(sum([i*j for (i, j) in zip(window, weights)]), 2)
        weighted_moving_averages.append(window_average)	
        i += 1
    return weighted_moving_averages

def calculate_metrics(y_true, y_pred, classes):
    """
    Calculate precision, recall, and accuracy for two lists.
    """
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

    # Calculate average precision, recall, and F1-score
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_accuracy = accuracy

    return precision, recall, accuracy, avg_precision, avg_recall, avg_accuracy

def plot_confusion_matrix(y_true, y_pred, classes, tar_file, title):
    """
    Calculate and plot the confusion matrix for two lists.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Add labels to the plot
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label',
           ylabel='True label')
    '''
    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    '''
    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Add a title to the plot
    ax.set_title(title)

    # Show the plot
    plt.savefig(tar_file)
    plt.close()



model_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs'
root_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction'

posture2d_root = os.path.join(model_fd, 'posture_2d_res')
vid_list = os.listdir(posture2d_root)
print(vid_list)

posture3d_root = os.path.join(model_fd, 'posture_3d_res')

bbox_root = os.path.join(root_fd, 'bbox_data')


tar_root = model_fd
tar_path = os.path.join(tar_root, 'signal_comparison')

train_vid = np.load('/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset/train_videos.npy',allow_pickle=True)

test_vid = np.load('/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset/test_videos.npy',allow_pickle=True)

rest_vid = np.load('/work/aclab/xiaof.huang/fu.n/InfantAction/InfAct_images_dataset/rest_videos.npy',allow_pickle=True)

train_vid = np.concatenate((train_vid, rest_vid), axis=0)


action_map = {'1':[0,1],'2':[0,2],'3':[0,3],'4':[0,4],'5':[1,0],'6':[1,2],'7':[1,3],'8':[1,4],'9':[2,0],'10':[2,1],'11':[2,3],'12':[2,4],'13':[3,0],'14':[3,1],'15':[3,2],'16':[3,4],'17':[4,0],'18':[4,1],'19':[4,2],'20':[4,3]}
keylist = list(action_map.keys())
valuelist = list(action_map.values())

pred_2d_action_list = []
pred_3d_action_list = []
vid_name_list = []
pred_2d_posture_list = []
pred_3d_posture_list = []

w_len = 8

entire_gt_action_list = list(test_vid[:, 1:2].flatten())
print(len(entire_gt_action_list))
gt_action_list = []
for i in range(test_vid.shape[0]):
    if int(test_vid[i][4]) < 25 or int(test_vid[i][6]) < 25:
        print('invalid!')
        print(str(int(test_vid[i][0])))
        continue
        
    gt_action_list.append(entire_gt_action_list[i])
    vid_name = str(int(test_vid[i][0]))
    vid_name_list.append(int(test_vid[i][0]))

    obj_scores = []
    len_str = '04d'
    vid_bbox = os.path.join(bbox_root, vid_name + '/' + vid_name)
    files_list = os.listdir(vid_bbox)
    for idx in range(int(len(files_list)/2)):
        frame_name = 'frame' + format(idx, len_str) + '.txt'
        frame_file = os.path.join(vid_bbox, frame_name)
        if os.path.exists(frame_file):
            with open(frame_file) as bbox_f:
                bbox_data = [(line.strip()) for line in bbox_f.readlines()]
            bbox_list = [float(i) for i in bbox_data[0].split()] 
            #print(bbox_list)
            obj_scores.append(bbox_list[5])
        else:
            obj_scores.append(0.1)


    output1_file = os.path.join(posture2d_root, vid_name, 'pred.npy')
    pred1 = np.load(output1_file,allow_pickle=True)
    pred1 = [element.cpu().numpy() for sublist in pred1 for element in sublist]
    pred1 = np.array(pred1).flatten()
 
    scores1 = np.load(os.path.join(posture2d_root, vid_name,'score.npy'),allow_pickle=True)
    score1 = scores1[0]
    for ele in range(1, len(scores1)):
       score1 = np.concatenate((score1, scores1[ele]))
    
    start_sq1 = pred1[:int(test_vid[i][4])]
    end_sq1 = pred1[(int(test_vid[i][4])+int(test_vid[i][5])):int(test_vid[i][7])]

    start_sig_sq1 = score1[:int(test_vid[i][4]),:]
    end_sig_sq1 = score1[(int(test_vid[i][4])+int(test_vid[i][5])):int(test_vid[i][7]),:]
    
    start_obj_sq = obj_scores[:int(test_vid[i][4])]
    end_obj_sq = obj_scores[(int(test_vid[i][4])+int(test_vid[i][5])):int(test_vid[i][7])]
    '''

    
    start_sq1 = pred1[:25]
    end_sq1 = pred1[-25:]
    print('3333333333333333333333333')
    print(len(start_sq1))
    print(len(end_sq1))
    start_sig_sq1 = score1[:25,:]
    end_sig_sq1 = score1[-25:,:]
    
    start_obj_sq = obj_scores[:25]
    end_obj_sq = obj_scores[-25:]
    '''

    #vis_signal(start_sig_sq1, end_sig_sq1,'2D before BWMA', vid_name, tar_path)
    
    # refinement methods
    window_len = w_len
    '''
    #moving average smoothing
    for k in range(5):
        start_sig_sq1[:, k] = np.transpose(moving_average(list(start_sig_sq1[:,k]), window_len))
        end_sig_sq1[:, k] = np.transpose(moving_average(list(end_sig_sq1[:,k]), window_len))
    '''
    '''
    #boundingbox weighted moving average smoothing
    for k in range(5):
        start_sig_sq1[:, k] = np.transpose(obj_weighted_moving_average(list(start_sig_sq1[:,k]), start_obj_sq, window_len))
        end_sig_sq1[:, k] = np.transpose(obj_weighted_moving_average(list(end_sig_sq1[:,k]), end_obj_sq, window_len))
    '''
    '''
    #exponentially weighted moving average smoothing
    df_start_sig1 = pd.DataFrame(start_sig_sq1,columns=['0','1','2','3','4'])
    df_end_sig1 = pd.DataFrame(end_sig_sq1,columns=['0','1','2','3','4'])

    ewm_start_sig1 = df_start_sig1.ewm(alpha=2 / 3).mean()
    ewm_end_sig1 = df_end_sig1.ewm(alpha=2 / 3).mean()
    start_sig_sq1 = ewm_start_sig1.to_numpy()
    end_sig_sq1 = ewm_end_sig1.to_numpy()
    '''

    #vis_signal(start_sig_sq1, end_sig_sq1,'2D after BWMA', vid_name, tar_path)

    start_sq1 = np.argmax(start_sig_sq1, axis=1) 
    end_sq1 = np.argmax(end_sig_sq1, axis=1) 


    posture_start1 = stats.mode(start_sq1)[0][0]
    posture_end1 = stats.mode(end_sq1)[0][0]
    tmp = [posture_start1, posture_end1]
    pred_2d_posture_list.append(posture_start1)
    pred_2d_posture_list.append(posture_end1)

    for j in range(len(keylist)):
        if valuelist[j] == tmp:
            pred_2d_action_list.append(int(keylist[j]))
    
    if posture_start1 == posture_end1:
        pred_2d_action_list.append(0)

        

    output2_file = os.path.join(posture3d_root, vid_name, 'pred.npy')
    pred2 = np.load(output2_file,allow_pickle=True)
    pred2 = [element.cpu().numpy() for sublist in pred2 for element in sublist]
    pred2 = np.array(pred2).flatten()
    
    scores2 = np.load(os.path.join(posture3d_root, vid_name,'score.npy'),allow_pickle=True)
    score2 = scores2[0]
    for ele in range(1, len(scores2)):
       score2 = np.concatenate((score2, scores2[ele]))
    
    start_sq2 = pred2[:int(test_vid[i][4])]
    end_sq2 = pred2[(int(test_vid[i][4])+int(test_vid[i][5])):int(test_vid[i][7])]

    start_sig_sq2 = score2[:int(test_vid[i][4]),:]
    end_sig_sq2 = score2[(int(test_vid[i][4])+int(test_vid[i][5])):int(test_vid[i][7]),:]
    '''

    start_sq2 = pred1[:25]
    end_sq2 = pred1[-25:]
    
    start_sig_sq2 = score1[:25,:]
    end_sig_sq2 = score1[-25:,:]
    
    '''
    #vis_signal(start_sig_sq2, end_sig_sq2,'3D before BWMA', vid_name, tar_path)

    # refinement methods
    window_len = w_len
    '''
    #moving average smoothing
    for k in range(5):
        start_sig_sq2[:, k] = np.transpose(moving_average(list(start_sig_sq2[:,k]), window_len))
        end_sig_sq2[:, k] = np.transpose(moving_average(list(end_sig_sq2[:,k]), window_len))
    '''
    '''
    #boundingbox weighted moving average smoothing
    for k in range(5):
        start_sig_sq2[:, k] = np.transpose(obj_weighted_moving_average(list(start_sig_sq2[:,k]), start_obj_sq, window_len))
        end_sig_sq2[:, k] = np.transpose(obj_weighted_moving_average(list(end_sig_sq2[:,k]), end_obj_sq, window_len))
    '''
    '''
    #exponentially weighted moving average smoothing
    df_start_sig2 = pd.DataFrame(start_sig_sq2,columns=['0','1','2','3','4'])
    df_end_sig2 = pd.DataFrame(end_sig_sq2,columns=['0','1','2','3','4'])

    ewm_start_sig2 = df_start_sig2.ewm(alpha=2 / 3).mean()
    ewm_end_sig2 = df_end_sig2.ewm(alpha=2 / 3).mean()
    start_sig_sq2 = ewm_start_sig2.to_numpy()
    end_sig_sq2 = ewm_end_sig2.to_numpy()
    '''

    #vis_signal(start_sig_sq2, end_sig_sq2,'3D after BWMA', vid_name, tar_path)


    start_sq2 = np.argmax(start_sig_sq2, axis=1) 
    end_sq2 = np.argmax(end_sig_sq2, axis=1)



    posture_start2 = stats.mode(start_sq2)[0][0]
    posture_end2 = stats.mode(end_sq2)[0][0]
    tmp = [posture_start2, posture_end2]
    pred_3d_posture_list.append(posture_start2)
    pred_3d_posture_list.append(posture_end2)


    for j in range(len(keylist)):
        if valuelist[j] == tmp:
            pred_3d_action_list.append(int(keylist[j]))

    if posture_start2 == posture_end2:
        pred_3d_action_list.append(0)


print(len(gt_action_list))
print(len(pred_2d_action_list))
print(len(pred_3d_action_list))
print(pred_3d_posture_list)

hr_dict = dict (vid_names = vid_name_list, gt = gt_action_list, pred_2d = pred_2d_action_list, pred_3d = pred_3d_action_list)

# Create DataFrame from Dictionary
hr_df2 = pd.DataFrame(hr_dict, columns = ['vid_names', 'gt', 'pred_2d', 'pred_3d'] )
print(hr_df2)

#export to csv format
hr_df2.to_csv('/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs/comp.csv', index = False)


pred_dict = dict (pos_2d = pred_2d_posture_list, pos_3d = pred_3d_posture_list)
pred_df2 = pd.DataFrame(pred_dict, columns =['pred_2d', 'pred_3d'] )
pred_df2.to_csv('/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs/comp_pose.csv', index = False)


classes = ['1','5','8','9','10','11','15','16','19']

precision, recall, accuracy, avg_precision, avg_recall, avg_accuracy = calculate_metrics(gt_action_list, pred_2d_action_list, classes)

# Print the results
print("Per-class Metrics:")
for i, c in enumerate(classes):
    print(f"{c}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}\n")
print(f"Overall Metrics:")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Overall Accuracy: {avg_accuracy}")

'''
# Plot the confusion matrix
tar_file = os.path.join(tar_root, 'cm_ft_posture_2d_test_video_frames.png')
title = "Finetuned 2D Pose-based Posture"
plot_confusion_matrix(gt_list, pred_2d_list, classes, tar_file, title)
'''

precision, recall, accuracy, avg_precision, avg_recall, avg_accuracy = calculate_metrics(gt_action_list, pred_3d_action_list, classes)

# Print the results
print("Per-class Metrics:")
for i, c in enumerate(classes):
    print(f"{c}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}\n")
print(f"Overall Metrics:")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
print(f"Overall Accuracy: {avg_accuracy}")

'''
# Plot the confusion matrix
tar_file = os.path.join(tar_root, 'cm_ft_posture_3d_test_video_frames.png')
title = "Finetuned 3D Pose-based Posture"
plot_confusion_matrix(gt_list, pred_3d_list, classes, tar_file, title)
'''
