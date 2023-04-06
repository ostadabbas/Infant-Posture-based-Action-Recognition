import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.patches as patches
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from matplotlib import rc
from matplotlib.patches import Rectangle
import pandas as pd

video_score = []
obj_root1 = '/work/aclab/xiaof.huang/fu.n/InfantAction/version5_183_outputs/bbox_data'
obj_root2 = '/work/aclab/xiaof.huang/fu.n/InfantAction/bbox_data'

ave_score = []
for i in range(209):
    vid_idx = i + 1
    if vid_idx <= 183:
        obj_file = os.path.join(os.path.join(obj_root1, str(vid_idx)), 'results.json')
    else:
        obj_file = os.path.join(os.path.join(obj_root2, str(vid_idx)), 'results.json')
    print(obj_file)
    f = open(obj_file)
    res = json.load(f)
    sum = 0
    num = len(res)
    for key, val in res.items():
        sum = sum + val[4]
    ave_score.append(sum/num)

plt.subplot(2, 1, 1)
plt.hist(np.array(ave_score))

x = np.arange(1, 210)
plt.subplot(2, 1, 2)
plt.plot(x, np.array(ave_score), color ='maroon')
#plt.show() 
plt.savefig('/work/aclab/xiaof.huang/fu.n/InfantAction/hist_bbox_ave_score.png')

# 95% threshold
print('Threshold 95%:')
idx = np.where(np.array(ave_score)<0.95)
print('index ', idx)
print('score', np.array(ave_score)[np.array(ave_score)<0.95])

# 90% threshold
print('Threshold 90%:')
idx = np.where(np.array(ave_score)<0.90)
print('index ', idx)
print('score ', np.array(ave_score)[np.array(ave_score)<0.90])

# 85% threshold
print('Threshold 85%:')
idx = np.where(np.array(ave_score)<0.85)
print('index ', idx)
print('score ', np.array(ave_score)[np.array(ave_score)<0.85])
