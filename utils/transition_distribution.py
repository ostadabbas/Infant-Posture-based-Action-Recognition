import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from io import StringIO

# read 8 transitional action groundtruth labels
df = pd.read_excel('./infantactionlabel.xlsx')
data = df.to_numpy()
gt_action = data[:,3]
print(gt_action.shape)

# read 4 posture groundtruth labels
df1 = pd.read_csv('./posture_list_700.csv', dtype = str, header=None)
data1 = df1.to_numpy()
posture = []
for i in range(data1.shape[0]):
    if data1[i, 0] == 'Supine':
        posture.append(1)
    elif data1[i, 0] == 'Prone':
        posture.append(2)
    elif data1[i, 0] == 'Sitting':
        posture.append(3)
    elif data1[i, 0] == 'Standing':
        posture.append(4)

gt_posture = np.array(posture)
print(gt_posture.shape)


# read posture prediction 
imgs = np.load('/home/faye/Documents/FiDIP_Posture/Classifier/test_kpts_output_202207241440/img.npy',allow_pickle=True)
pred = np.load('/home/faye/Documents/FiDIP_Posture/Classifier/test_kpts_output_202207241440/pred.npy',allow_pickle=True)    
scores = np.load('/home/faye/Documents/FiDIP_Posture/Classifier/test_kpts_output_202207241440/score.npy',allow_pickle=True)
pred = pred[0]
imgs = imgs[0]
pred_posture = scores[0]


posture_label = ['Supine', 'Prone', 'Sitting', 'Standing']

trans2_dist = np.zeros((6, 1))   # order: 1<->2, 1<->3, 1<->4, 2<->3, 2<->4, 3<->4
for index in range(len(imgs)):
    print(index)
    img_name = imgs[index]
    print(img_name)
    posture_p = pred[index]
    score_p = pred_posture[index]

    if gt_action[index] == 9:
        continue

    idx_list = np.argsort(score_p)
    posture1 = idx_list[3]
    posture2 = idx_list[2]

    if (posture1 == 0 and posture2 == 1) or (posture1 == 1 and posture2 == 0):
        trans2_dist[0, 0] = trans2_dist[0, 0] + 1
    elif (posture1 == 0 and posture2 == 2) or (posture1 == 2 and posture2 == 0):
        trans2_dist[1, 0] = trans2_dist[1, 0] + 1
    elif (posture1 == 0 and posture2 == 3) or (posture1 == 3 and posture2 == 0):
        trans2_dist[2, 0] = trans2_dist[2, 0] + 1
    elif (posture1 == 1 and posture2 == 2) or (posture1 == 2 and posture2 == 1):
        trans2_dist[3, 0] = trans2_dist[3, 0] + 1
    elif (posture1 == 1 and posture2 == 3) or (posture1 == 3 and posture2 == 1):
        trans2_dist[4, 0] = trans2_dist[4, 0] + 1
    elif (posture1 == 2 and posture2 == 3) or (posture1 == 3 and posture2 == 2):
        trans2_dist[5, 0] = trans2_dist[5, 0] + 1

print(np.sum(trans2_dist))
fig = plt.figure(figsize = (8, 4))
class_list = ['Supine-Prone', 'Supine-Sitting', 'Supine-Standing', 'Prone-Sitting', 'Prone-Standing', 'Sitting-Standing']

plt.bar(class_list, list(trans2_dist[:, 0]), color ='b', width = 0.3)

plt.xlabel("Posture Class")
plt.ylabel("No. of transitional postures")
plt.show()


'''
print('Total Number of Transitional Postures: ' + str(np.count_nonzero(gt_action < 9)))

trans_dist = np.zeros((4, 1))
for j in range(len(gt_posture)):
    if gt_posture[j] == 1 and gt_action[j] != 9:
        trans_dist[0, 0] = trans_dist[0, 0] + 1
    elif gt_posture[j] == 2 and gt_action[j] != 9:
        trans_dist[1, 0] = trans_dist[1, 0] + 1
    elif gt_posture[j] == 3 and gt_action[j] != 9:
        trans_dist[2, 0] = trans_dist[2, 0] + 1
    elif gt_posture[j] == 4 and gt_action[j] != 9:
        trans_dist[3, 0] = trans_dist[3, 0] + 1

print(np.sum(trans_dist))
fig = plt.figure(figsize = (8, 4))
class_list = ['Supine', 'Prone', 'Sitting', 'Standing']

plt.bar(class_list, list(trans_dist[:, 0]), color ='b', width = 0.3)

plt.xlabel("Posture Class")
plt.ylabel("No. of transitional postures")
plt.show()
'''





        

