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
def get_arguments():
    parser = argparse.ArgumentParser(description="PCA")
    parser.add_argument("--start", type=int, default=0,
                        help="start pca layer.") 
    parser.add_argument("--end", type=int, default=15,
                        help="end pca layer.") 
    parser.add_argument("--root", type=str, default='/path/to/feat',
                        help="root for extracted features.")
    return parser.parse_args()
args = get_arguments()

channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096]

exact_list = ['adpater1_1', 'adpater1_2', 'adpater2_1', 'adpater2_2', 'adpater3_1', 'adpater3_2', 'adpater3_3', 
              'adpater4_1', 'adpater4_2', 'adpater4_3', 'adpater5_1', 'adpater5_2', 'adpater5_3', 'adpater6', 'adpater7', ]


for k in range(args.start, args.end):
    dim = channels[k]
    
    dir_feat = osp.join(args.root, exact_list[k] + '_wh')
    files = os.listdir(dir_feat)

    for i in range(len(files)):
        if i == 0:
            num = np.load(osp.join(dir_feat, files[i])).shape[0]
            features = np.zeros((len(files)*num, dim))
        features[i*num:(i+1)*num] = np.load(osp.join(dir_feat, files[i]))
    pca = decomposition.PCA(n_components=dim)
    pca.fit(features)
    pickle.dump(pca, open(osp.join(args.root, exact_list[k] + '_wh') + '_pca.sav', 'wb'))
    
    dir_feat = osp.join(args.root, exact_list[k] + '_rc')
    files = os.listdir(dir_feat)
    features = np.zeros((len(files)*num, dim))
    for i in range(len(files)):
        features[i*num:(i+1)*num] = np.load(osp.join(dir_feat, files[i]))
    pca = decomposition.PCA(n_components=dim)
    pca.fit(features)
    pickle.dump(pca, open(osp.join(args.root, exact_list[k] + '_rc') + '_pca.sav', 'wb'))    
