import numpy as np
import json
import os

# bbox expansion factor
scaleFactor = 1.2


tmp_file = '/work/aclab/xiaof.huang/fu.n/SPIN_infant_new/data/dataset_extras/SyRIP_valid.npz'
tmp = np.load(tmp_file)
centers = tmp['center']
imgnames = tmp['imgname']
parts = tmp['part']
scales = tmp['scale']

print(len(centers))
print(len(parts[0]))


pose2d_data = '/work/aclab/xiaof.huang/fu.n/InfantAction/custom_syrip'
tar_root = '/work/aclab/xiaof.huang/fu.n/InfantAction/videos_npz'
vid_list = os.listdir(pose2d_data)
for i in range(len(vid_list)):
    new_centers = []
    new_imgnames = []
    new_parts = []
    new_scales = []


    vid_name = vid_list[i]
    
    vid_folder = os.path.join(tar_root, vid_name)
    if not os.path.isdir(vid_folder):
        os.mkdir(vid_folder)

    anno_file = os.path.join(pose2d_data, vid_name, 'annotations', 'person_keypoints_validate_infant.json')
    f = open(anno_file)
    anno_data = json.load(f)
    annos = anno_data['annotations']
    images = anno_data['images']

    for j in range(len(annos)):
        imgname = 'test' + str(j) + '.jpg'
        w = images[j]['width']
        h = images[j]['height']
        new_imgnames.append(imgname)
        bbox = annos[j]['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

        new_centers.append(center)
        new_parts.append(list(np.zeros([24, 3])))
        new_scales.append(scale)
    
    outfile = os.path.join(vid_folder, 'video.npz')
    np.savez(outfile, center=new_centers, imgname=new_imgnames, part=new_parts, scale=new_scales)    


    

