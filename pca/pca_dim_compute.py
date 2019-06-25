import numpy as np
import random
import matplotlib.pyplot as plt
import os
import shutil
from sklearn import mixture
from sklearn import svm
from sklearn import decomposition
import pickle
import os.path as osp
import argparse

channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096]

exact_list = ['adpater1_1', 'adpater1_2', 'adpater2_1', 'adpater2_2', 'adpater3_1', 'adpater3_2', 'adpater3_3', 
              'adpater4_1', 'adpater4_2', 'adpater4_3', 'adpater5_1', 'adpater5_2', 'adpater5_3', 'adpater6', 'adpater7', ]


def PCA_Dim_Compute(root, thres):

    pca_dim, pca_dir = [], {}
    for k in range(0, 15):
        dim = channels[k]
        
        dir_feat = osp.join(root, exact_list[k] + '_wh')
        pca = pickle.load(open(dir_feat + '_pca.sav', 'rb'))
        ratio = pca.explained_variance_ratio_
        cum = [np.sum(ratio[0:i+1]) for i in range(dim)]
        cum_dim = np.sum(np.array(cum)<thres)
        if cum_dim < 2:
            cum_dim = 2

        pca_dim.append(cum_dim)
        pca_dir[osp.join(root, exact_list[k] + '_wh')] = osp.join(root, exact_list[k] + '_wh') + '_pca.sav'
        
        dir_feat = osp.join(root, exact_list[k] + '_rc')
        pca = pickle.load(open(dir_feat + '_pca.sav', 'rb'))
        ratio = pca.explained_variance_ratio_
        cum = [np.sum(ratio[0:i+1]) for i in range(dim)]
        cum_dim = np.sum(np.array(cum)<thres)
        if cum_dim < 2:
            cum_dim = 2   

        pca_dim.append(cum_dim)
        pca_dir[osp.join(root, exact_list[k] + '_rc')] = osp.join(root, exact_list[k] + '_rc') + '_pca.sav'
    

    for i in range(15):
        acc_dim = np.min((pca_dim[2*i], pca_dim[2*i+1])) * channels[i] * 2

    print 'pca threshold %5f, param ratio compared to residual adapter %5f' %(thres, acc_dim/35364864.0)
    return pca_dim, pca_dir
