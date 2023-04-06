import numpy as np
import json
import math
import matplotlib.pyplot as plt
 
gt_mimm_train = '/home/faye/Documents/InfantProject/data/MIMM_new/annotations/train_834/person_keypoints_train_infant.json'
gt_mimm_validate = '/home/faye/Documents/InfantProject/data/MIMM_new/annotations/validate_216/person_keypoints_validate_infant.json'

with open(gt_mimm_train, 'r') as fb1:
    d_gt_mimm_train = json.load(fb1)
    
with open(gt_mimm_validate, 'r') as fb2:
    d_gt_mimm_val = json.load(fb2)

def getPosture(d_gt):
    posture_list = []
    for i in range(len(d_gt['images'])):
        posture_list.append(d_gt['images'][i]['posture'])
    return posture_list
   
posture_mimm_train = getPosture(d_gt_mimm_train)
posture_mimm_validate = getPosture(d_gt_mimm_val)


def getDict(posture_list):
    a=b=c=d=e=0
    posture_dict = {'Supine', 'Prone', 'Sitting', 'Standing','None'}
    for i in range(len(posture_list)):
        if posture_list[i] == 'Supine':
            a += 1
        if posture_list[i] == 'Prone':
            b += 1
        if posture_list[i] == 'Sitting':
            c += 1
        if posture_list[i] == 'Standing':
            d += 1
        if posture_list[i] == 'Unknown':
            e += 1
        if posture_list[i] == 'None':
            e += 1    
    posture_dict = {'Supine': a, 'Prone':b, 'Sitting':c, 'Standing':d, 'None': e}
    return posture_dict

mimm_train = getDict(posture_mimm_train)
mimm_val = getDict(posture_mimm_validate)
   
def showBar(dist):
    csfont = {'fontname':'Times New Roman'}
    plt.bar(dist.keys(), dist.values(),width = 0.5)
    # plt.xticks(rotation=-30)

    for a,b in zip(dist.keys(), dist.values()):
        plt.text(a, b+0.01, '%.0f' % b, ha='center', va= 'bottom',**csfont,fontsize=12)
        plt.ylim(0,400)
        plt.xticks(**csfont,fontsize = 14)
        plt.yticks(**csfont,fontsize = 14)
        plt.xlabel('Posture Class',**csfont,fontsize = 16)
        plt.ylabel('The Number of Samples', **csfont,fontsize = 16)
        plt.title('Posture Distribution on mini-MIMM Dataset', **csfont,fontsize = 16)
        plt.show()
'''
showBar(mimm_train)

def showBarDouble(dict1, dict2):
    csfont = {'fontname':'Times New Roman'}
    plt.figure(figsize=(12,6)) 
    name = dict1.keys()
    value1 = dict1.values()
    value2 = dict2.values()
    x = range(4)
    bar1 = plt.bar(x, value1, width = 0.4,alpha = 0.8, color = 'r',label = 'train'
    for i in range(len(x)):
        x[i] = x[i] + 0.8
    bar2 = plt.bar(x, value2,width = 0.4,alpha = 0.8,color = 'g',label = 'validate') 
    plt.xticks(range(4),name)  
    plt.ylim(0,500)\n",
    plt.title('Posture Distribution',fontsize = 16)
    plt.xlabel('Posture Class',fontsize = 16)
    plt.ylabel('Number',fontsize = 16)
    plt.legend()
''' 
import pandas
import matplotlib.font_manager
 
def showBarDouble(dict1,dict2):
    csfont = {'fontname':'Times New Roman'}
    index = ['Supine', 'Prone', 'Sitting', 'Standing', 'None']
    data = np.zeros((len(index),2))
    data[0,0] = dict1['Supine']
    data[1,0] = dict1['Prone']
    data[2,0] = dict1['Sitting']
    data[3,0] = dict1['Standing']
    data[4,0] = dict1['None']
    print(dict1)
    data[0,1] = dict2['Supine']
    data[1,1] = dict2['Prone']
    data[2,1] = dict2['Sitting']
    data[3,1] = dict2['Standing']
    data[4,1] = dict2['None']
    print(dict2)
    pd = pandas.DataFrame(data, index=index,columns=['Train','Test'])
    pd.plot(kind='bar',ylim=[0,300],rot=0)
    idx = [0,1,2,3,4]
    for a,b in zip(idx,data[:,0]):
        plt.text(a-0.1,b+0.1, int(b),ha = 'center',va = 'bottom',**csfont,fontsize=12)
    for a,b in zip(idx,data[:,1]):
        plt.text(a+0.1,b+0.1, int(b),ha = 'center',va = 'bottom',**csfont,fontsize=12)
    plt.xticks(**csfont,fontsize = 14)
    plt.yticks(**csfont,fontsize = 14)     
    plt.xlabel('Posture Class',**csfont,fontsize = 16)
    plt.ylabel('The Number of Samples',**csfont,fontsize = 16)
    plt.legend(['Train','Test'],prop={'family': 'Times New Roman'}, fontsize = 16)
    plt.title('Posture Distribution on mini-MIMM Dataset',**csfont,fontsize = 16)
    plt.show()

showBarDouble(mimm_train, mimm_val)

