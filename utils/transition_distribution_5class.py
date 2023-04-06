import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from io import StringIO

# read 8 transitional action groundtruth labels and read 4 posture groundtruth labels
df = pd.read_excel('./infantactionlabel_v2.xlsx')
data = df.to_numpy()
gt_action = data[:,5]
gt_posture = data[:,3]
print(gt_action.shape)


# read posture prediction 
imgs = np.load('/home/faye/Documents/FiDIP_Posture/Classifier/test_syrip_kpts_output_202207310730/img.npy',allow_pickle=True)
pred = np.load('/home/faye/Documents/FiDIP_Posture/Classifier/test_syrip_kpts_output_202207310730/pred.npy',allow_pickle=True)    
scores = np.load('/home/faye/Documents/FiDIP_Posture/Classifier/test_syrip_kpts_output_202207310730/score.npy',allow_pickle=True)
pred = pred[0]
imgs = imgs[0]
pred_posture = scores[0]


posture_label = ['Supine', 'Prone', 'Sitting', 'Standing', 'All Fours']

trans2_dist = np.zeros((10, 1))   # order: 1<->2, 1<->3, 1<->4, 1<->5, 2<->3, 2<->4, 2<->5, 3<->4, 3<->5, 4<->5
for index in range(len(imgs)):
    print(index)
    img_name = imgs[index]
    print(img_name)
    posture_p = pred[index]
    score_p = pred_posture[index]

    if gt_action[index] == 9:
        continue

    idx_list = np.argsort(score_p)
    print(idx_list)
    posture1 = idx_list[3]
    posture2 = idx_list[2]

    if (posture1 == 0 and posture2 == 1) or (posture1 == 1 and posture2 == 0):
        trans2_dist[0, 0] = trans2_dist[0, 0] + 1
    elif (posture1 == 0 and posture2 == 2) or (posture1 == 2 and posture2 == 0):
        trans2_dist[1, 0] = trans2_dist[1, 0] + 1
    elif (posture1 == 0 and posture2 == 3) or (posture1 == 3 and posture2 == 0):
        trans2_dist[2, 0] = trans2_dist[2, 0] + 1
    elif (posture1 == 0 and posture2 == 4) or (posture1 == 4 and posture2 == 0):
        trans2_dist[3, 0] = trans2_dist[3, 0] + 1
    elif (posture1 == 1 and posture2 == 2) or (posture1 == 2 and posture2 == 1):
        trans2_dist[4, 0] = trans2_dist[4, 0] + 1
    elif (posture1 == 1 and posture2 == 3) or (posture1 == 3 and posture2 == 1):
        trans2_dist[5, 0] = trans2_dist[5, 0] + 1
    elif (posture1 == 1 and posture2 == 4) or (posture1 == 4 and posture2 == 1):
        trans2_dist[6, 0] = trans2_dist[6, 0] + 1
    elif (posture1 == 2 and posture2 == 3) or (posture1 == 3 and posture2 == 2):
        trans2_dist[7, 0] = trans2_dist[7, 0] + 1
    elif (posture1 == 2 and posture2 == 4) or (posture1 == 4 and posture2 == 2):
        trans2_dist[8, 0] = trans2_dist[8, 0] + 1
    elif (posture1 == 3 and posture2 == 4) or (posture1 == 4 and posture2 == 3):
        trans2_dist[9, 0] = trans2_dist[9, 0] + 1

print(np.sum(trans2_dist))
fig = plt.figure(figsize = (8, 4))
class_list = ['Supine-Prone', 'Supine-Sitting', 'Supine-Standing', 'Suping-AllFours', 'Prone-Sitting', 'Prone-Standing', 'Prone-AllFours', 'Sitting-Standing', 'Sitting-AllFours', 'Standing-AllFours']

plt.bar(class_list, list(trans2_dist[:, 0]), color ='b', width = 0.3)

plt.xlabel("Posture Class")
plt.ylabel("No. of transitional postures")
plt.show()



print('Total Number of Transitional Postures: ' + str(np.count_nonzero(gt_action < 9)))

trans_dist = np.zeros((5, 1))
posture_dist = np.zeros((5, 1))
for j in range(len(gt_posture)):
    if gt_posture[j] == 1 and gt_action[j] != 9:
        trans_dist[0, 0] = trans_dist[0, 0] + 1
    elif gt_posture[j] == 2 and gt_action[j] != 9:
        trans_dist[1, 0] = trans_dist[1, 0] + 1
    elif gt_posture[j] == 3 and gt_action[j] != 9:
        trans_dist[2, 0] = trans_dist[2, 0] + 1
    elif gt_posture[j] == 4 and gt_action[j] != 9:
        trans_dist[3, 0] = trans_dist[3, 0] + 1
    elif gt_posture[j] == 5 and gt_action[j] != 9:
        trans_dist[4, 0] = trans_dist[4, 0] + 1

    if gt_posture[j] == 1:
        posture_dist[0, 0] = posture_dist[0, 0] + 1
    elif gt_posture[j] == 2:
        posture_dist[1, 0] = posture_dist[1, 0] + 1
    elif gt_posture[j] == 3:
        posture_dist[2, 0] = posture_dist[2, 0] + 1
    elif gt_posture[j] == 4:
        posture_dist[3, 0] = posture_dist[3, 0] + 1
    elif gt_posture[j] == 5:
        posture_dist[4, 0] = posture_dist[4, 0] + 1

print(np.sum(trans_dist))
fig = plt.figure(figsize = (8, 4))
class_list = ['Supine', 'Prone', 'Sitting', 'Standing', 'All Fours']

X_axis = np.arange(len(class_list))

plt.bar(X_axis - 0.1, list(trans_dist[:, 0]), 0.2, label = 'Transitional')
plt.bar(X_axis + 0.1, list(posture_dist[:, 0]), 0.2, label = 'Total')
plt.xticks(X_axis, class_list)
plt.xlabel("Posture Class")
plt.ylabel("No. of transitional postures")
plt.legend()
plt.show()






        

