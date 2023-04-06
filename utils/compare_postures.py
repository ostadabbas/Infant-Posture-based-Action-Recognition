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


pred1_folder = '/home/faye/Documents/InfantProject/outputs/example1_outputs/example1_posture_result_raw'
tar1 = np.load(os.path.join(pred1_folder, 'tar.npy'),allow_pickle=True)
imgs1 = np.load(os.path.join(pred1_folder, 'img.npy'),allow_pickle=True)
pred1 = np.load(os.path.join(pred1_folder, 'pred.npy'),allow_pickle=True)    
scores1 = np.load(os.path.join(pred1_folder, 'score.npy'),allow_pickle=True)
tar1 = tar1[0]
pred1 = pred1[0]
imgs1 = imgs1[0]
scores1 = scores1[0]

pred2_folder = '/home/faye/Documents/InfantProject/outputs/example1_outputs/example1_posture_withoutTrans_result_raw'
tar2 = np.load(os.path.join(pred2_folder, 'tar.npy'),allow_pickle=True)
imgs2 = np.load(os.path.join(pred2_folder, 'img.npy'),allow_pickle=True)
pred2 = np.load(os.path.join(pred2_folder, 'pred.npy'),allow_pickle=True)    
scores2 = np.load(os.path.join(pred2_folder, 'score.npy'),allow_pickle=True)
tar2 = tar2[0]
pred2 = pred2[0]
imgs2 = imgs2[0]
scores2 = scores2[0]

t = np.arange(0, len(imgs1))

plt.subplot(2, 1, 1)
plt.plot(t/30, scores1[:, 0], c='r', label='Supine')
plt.plot(t/30, scores1[:, 1], c='b', label='Prone')
plt.plot(t/30, scores1[:, 2], c='g', label='Sitting')
plt.plot(t/30, scores1[:, 3], c='c', label='Standing')
plt.title('Posture Signal Inferred by Original Posture Classifier')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Confidence Score')

plt.subplot(2, 1, 2)
plt.plot(t/30, scores2[:, 0], c='r', label='Supine')
plt.plot(t/30, scores2[:, 1], c='b', label='Prone')
plt.plot(t/30, scores2[:, 2], c='g', label='Sitting')
plt.plot(t/30, scores2[:, 3], c='c', label='Standing')
plt.title('Posture Signal Inferred by New Posture Classifier')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Confidence Score')

plt.show()

