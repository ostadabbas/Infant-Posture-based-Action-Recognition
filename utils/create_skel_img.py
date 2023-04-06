import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import matplotlib.patches as patches
from matplotlib import colors
import json

# edges = [[5, 6], [5, 7],
#          [7, 9], [6, 8], [8, 10], [11, 12], [11, 13], [13, 15],
#          [12, 14], [14,16]]  # 17 keypoints format
         
edges = [[0, 1], [1, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 7],
         [1, 8], [8, 9], [8, 10], [10, 12], [9, 11], [11, 13],
         [12, 14]]  # 15 keypoints format
edges = np.array(edges)

def generate_color_map():
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 12)]
    color_map = []
    for c in colors:
        b = max(0, min(255, int(np.floor(c[0] * 256.0))))
        g = max(0, min(255, int(np.floor(c[1] * 256.0))))
        r = max(0, min(255, int(np.floor(c[2] * 256.0))))
        color_map.append((b, g, r))
    return color_map

def generate_skel_img(w, h, kpts, edges, color_map):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = 0
    num_joints = kpts.shape[0]
    num_edge = len(edges)
    plt.imshow(img)
    '''
    for i in range(num_edge):
        if kpts[edges[i, 0], 2] != 0 and kpts[edges[i, 1], 2] != 0:
            x0 = int(kpts[edges[i, 0], 0])
            x1 = int(kpts[edges[i, 1], 0])

            y0 = int(kpts[edges[i, 0], 1])
            y1 = int(kpts[edges[i, 1], 1])

            img = cv2.line(img, (x0, y0), (x1, y1), color = (255, 255, 255), thickness = 6)

    if kpts[5, 2] != 0 and kpts[6, 2] != 0 and kpts[11, 2] != 0 and kpts[12, 2] != 0:
        x0 = int((kpts[5, 0] + kpts[6, 0]) / 2)
        x1 = int((kpts[11, 0] + kpts[12, 0]) / 2)

        y0 = int((kpts[5, 1] + kpts[6, 1]) / 2)
        y1 = int((kpts[11, 1] + kpts[12, 1]) / 2)

        img = cv2.line(img, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=6)

    for a in range(num_joints - 5):
        a += 5
        if kpts[a, 2] != 0:
            img = cv2.circle(img, (int(kpts[a, 0]), int(kpts[a, 1])), 6, color_map[a - 5], -1)
    '''
    for i in range(num_edge):
        x0 = int(kpts[edges[i, 0], 0])
        x1 = int(kpts[edges[i, 1], 0])

        y0 = int(kpts[edges[i, 0], 1])
        y1 = int(kpts[edges[i, 1], 1])
        img = cv2.line(img, (x0, y0), (x1, y1), color = (0, 0, 255), thickness = 8)
    
    
    for a in range(num_joints):
        # if kpts[a,2] == 2:
        if a == 0:
            continue
        if a == 1:
            continue
        if a == 8:
            continue
        else:
            img = cv2.circle(img, (int(kpts[a, 0]), int(kpts[a, 1])), 6, color = (0, 215, 255), thickness = 8)
    
    
    # plt.axis('off')
    # plt.savefig(os.path.join(output, file_name),bbox_inches = 'tight',dpi = 300, pad_inches=0.0)
    # plt.close()
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Review Posture Annotation')
    parser.add_argument('-a', '--annotation_file', default='./miniMIMM/MIMM_keypoints_train.json', type=str)
    parser.add_argument('-o', '--output_folder', default='./bbox_test/output1/', type=str)
    opt = parser.parse_args()
    
    output_folder = './bbox_test/output1/'
    anno_path = opt.annotation_file
    output_path = opt.output_folder
    with open(anno_path) as json_file:
        anno = json.load(json_file)
    images = anno['images']
    annotations = anno['annotations']

    color_map = generate_color_map()
    
    img_name = []
    id_list = []
    w = []
    h = []
    kpts = np.zeros((len(annotations), 17, 3))
    for i in range(len(images)):
        w.append(images[i]['width'])
        h.append(images[i]['height'])
        img_name.append(images[i]['file_name'])
        # img_path = os.path.join(output_path, img_name)
        id_list.append(images[i]['id'])

        tmps = annotations[i]['keypoints']
        # kpts = np.zeros((int(len(tmps) / 3), 3))
        kpts[i, :,0] = tmps[0::3]
        kpts[i, :,1] = tmps[1::3]
        kpts[i, :,2] = tmps[2::3]
    
    tmp_list = np.load('hc.npy', allow_pickle=True)
    hc_list = [abc for ab in tmp_list for abc in ab]
    c = 0
    for i in range(len(kpts)):
        kpts[i,0,0] = hc_list[c]
        kpts[i,0,1] = hc_list[c+1]
        kpts[i,0,2] = hc_list[c+2]
        c += 3 

    keypoints = np.zeros((len(kpts),15,3))    
    for k in range(len(kpts)):
        keypoints[k,0,:] = kpts[k,0,:]
        keypoints[k,1,:] = (kpts[k,5,:] + kpts[k,6,:])/2
        keypoints[k,2:8,:] = kpts[k,5:11,:]
        keypoints[k,8,:] = (kpts[k,11,:] + kpts[k,12,:])/2
        keypoints[k,9:,:] = kpts[k,11:,:]
                   
    for pth, dir_list, file_list in os.walk(r'./bbox_test/input'):
        for file_name in file_list:
            print(file_name)
            for i in range(len(img_name)):
                if file_name == img_name[i]:
                    print('a')
                    img_path = os.path.join(pth, file_name)
                    output_pth = os.path.join(output_folder, file_name)
                    img = generate_skel_img(w[i], h[i], keypoints[i], edges, color_map)
                    cv2.imwrite(output_pth, img)
        #cv2.imshow(img_name, img)
        #cv2.waitKey(0)
