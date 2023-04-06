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
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix, precision_score, recall_score

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

#rc('text', usetex = True)
#rc('text.latex', preamble = '\usepackage{color}')

model_fd = '/work/aclab/xiaof.huang/fu.n/InfantAction/ft_posture_model_outputs'
posture2d_root = os.path.join(model_fd, 'posture_2d_res')
vid_list = os.listdir(posture2d_root)
print(vid_list)

posture3d_root = os.path.join(model_fd, 'posture_3d_res')

tar_root = model_fd

anno_file = '/work/aclab/xiaof.huang/fu.n/InfantAction/posture_model_inputs/test100_train300_2d/annotations/test100/person_keypoints_validate_infant_vidframes_5class.json'
f = open(anno_file,)
anno = json.load(f)
images = anno['images']

gt_list = []
pred_2d_list = []
pred_3d_list = []
vid_list = dict()
for i in range(len(images)):
    vid_name = images[i]['original_video_name']
    frame_idx = int(images[i]['original_file_name'][4:-4])
    
    if images[i]['posture'] == 'Supine':
        gt_list.append(0)
    elif images[i]['posture'] == 'Prone':
        gt_list.append(1)
    elif images[i]['posture'] == 'Sitting':
        gt_list.append(2)
    elif images[i]['posture'] == 'Standing':
        gt_list.append(3)
    elif images[i]['posture'] == 'All Fours':
        gt_list.append(4)


    pred1 = np.load(os.path.join(posture2d_root, vid_name + '/pred.npy'), allow_pickle=True) 
    pred1 = [element.cpu().numpy() for sublist in pred1 for element in sublist]
    ori_posture2d_pred = np.array(pred1).flatten()
    pred_2d_list.append(ori_posture2d_pred[frame_idx])
 
    pred2 = np.load(os.path.join(posture3d_root, vid_name + '/pred.npy'), allow_pickle=True) 
    pred2 = [element.cpu().numpy() for sublist in pred2 for element in sublist]
    ori_posture3d_pred = np.array(pred2).flatten()
    pred_3d_list.append(ori_posture3d_pred[frame_idx])

    vid_list[vid_name+'_'+images[i]['original_file_name'][4:-4]] = [gt_list[-1], ori_posture2d_pred[frame_idx], ori_posture3d_pred[frame_idx]]


print(len(pred_3d_list))
print(gt_list)
print(pred_2d_list)
print(pred_3d_list)
print(vid_list)
print(len(vid_list.keys()))

classes = ["Supine", "Prone", "Sitting", "Standing", "All-fours"]

precision, recall, accuracy, avg_precision, avg_recall, avg_accuracy = calculate_metrics(gt_list, pred_2d_list, classes)

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


# Plot the confusion matrix
tar_file = os.path.join(tar_root, 'cm_ori_posture_2d_test_video_frames.png')
title = "Original 2D Pose-based Posture"
plot_confusion_matrix(gt_list, pred_2d_list, classes, tar_file, title)

precision, recall, accuracy, avg_precision, avg_recall, avg_accuracy = calculate_metrics(gt_list, pred_3d_list, classes)
print(accuracy)

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


# Plot the confusion matrix
tar_file = os.path.join(tar_root, 'cm_ori_posture_3d_test_video_frames.png')
title = "Original 3D Pose-based Posture"
plot_confusion_matrix(gt_list, pred_3d_list, classes, tar_file, title)

