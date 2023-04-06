import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

feature_array_s = np.load('SVHN_USPS_s_feature_after.npy')  # source features
#img_label_s = np.load('USPS_MNIST_s_feature_before.npy') 
print(np.shape(feature_array_s))
img_label_s = np.zeros((np.shape(feature_array_s)[0],1))
#print(np.shape(img_label_s))


feature_array_t = np.load('SVHN_USPS_t_feature_after.npy')   # target features
print(np.shape(feature_array_t))
img_label_t = np.ones((np.shape(feature_array_t)[0],1))
'''
feature_array_coco = np.load('feature_array.npy')
print(np.shape(feature_array_coco))
img_label_coco = np.load('img_label.npy')
print(np.shape(img_label_coco))
'''
n_s = len(feature_array_s)
n_t = len(feature_array_t)
# feature_array = np.concatenate((feature_array_s[0:n_s:2], feature_array_t[0:n_t:5]), axis=0)
# img_label = np.concatenate((img_label_s[0:n_s:2], img_label_t[0:n_t:5]), axis=0)

feature_array = np.concatenate((feature_array_s, feature_array_t), axis=0)
img_label = np.concatenate((img_label_s, img_label_t), axis=0)

print(feature_array.shape)
avgpool = nn.AdaptiveAvgPool2d((1,1))

def scale_range(x):
    a = np.max(x)-np.min(x)
    b = x-np.min(x)
    return b/a


print('lalalallala: ', np.shape(feature_array))
feature_array = avgpool(torch.from_numpy(feature_array))
tmp = torch.flatten(feature_array,1)
tsne = TSNE(n_components = 2, init='pca').fit_transform(tmp) # (1004,2)
print(np.shape(tsne))

#tx = scale_range(tsne[:,0])
#ty = scale_range(tsne[:,1])
tx = tsne[:,0]
ty = tsne[:,1]

fig = plt.figure()
ax = fig.add_subplot(111)


classes = []
colors = []
target_num = 0 
source_num = 0


for idx in range(len(img_label)):
    if img_label[idx] == 1:
        classes.append('Target')
        colors.append('m')
        target_num += 1
    if img_label[idx] == 0:
        classes.append('Source')
        colors.append('c')
        source_num += 1
print('target number: ', target_num)
print('source number:', source_num)

for (i,cla) in enumerate(set(classes)):
    xc = [p for (j,p) in enumerate(tx) if classes[j]==cla]
    yc = [p for (j,p) in enumerate(ty) if classes[j]==cla]
    cols = [c for (j,c) in enumerate(colors) if classes[j]==cla]
    plt.scatter(xc,yc,c=cols,label=cla, s = 16, alpha = 0.6)
plt.legend(loc='upper right')

plt.savefig('SVHN_USPS_feature_after.png')
plt.show()


