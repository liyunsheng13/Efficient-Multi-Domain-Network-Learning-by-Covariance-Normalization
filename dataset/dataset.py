import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image


class DataSet(data.Dataset):
    def __init__(self, root, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = np.array([osp.join(self.root, osp.join('train_data', l.split(' ')[0])) for l in open(osp.join(self.root, 'TrainImages.txt'))])
        self.img_labels = np.array([np.int32(l.split(' ')[1]) for l in open(osp.join(self.root, 'TrainImages.txt'))])        
        self.files = []

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        
        name = self.img_ids[index]
        image = Image.open(name + '.jpg').convert('RGB')
        image = np.array(image, dtype=float)
        label = self.img_labels[index]

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
    
class TestDataSet(data.Dataset):
    def __init__(self, root, mean=(128, 128, 128)):
        self.root = root
        self.mean = mean
        self.img_ids = np.array([osp.join(self.root, osp.join('test_data', l.split(' ')[0])) for l in open(osp.join(self.root, 'TestImages.txt'))])
        self.img_labels = np.array([np.int32(l.split(' ')[1]) for l in open(osp.join(self.root, 'TestImages.txt'))])        
        

    def __len__(self):

        return len(self.img_ids)


    def __getitem__(self, index):

        name = self.img_ids[index]
        image = Image.open(name + '.jpg').convert('RGB')
        image = np.array(image, dtype=float)        
        label = self.img_labels[index]

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


