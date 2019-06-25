import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
from torchvision import models
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib
matplotlib.rcParams['backend'] = 'Qt5Agg'
matplotlib.rcParams['backend.qt5'] = 'PyQt5'
import matplotlib.pyplot as plt
import random
from model.vgg_adapter import VGG
from dataset.dataset_feat_extr import DataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
      
    parser.add_argument("--num-workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default='/path/to/dataset',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--snapshot-dir", type=str, default='/path/to/checkpoints/',
                        help="Where to save snapshots of the model.")    
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--start", type=int, default=0,
                        help="Number of epoch.") 
    parser.add_argument("--end", type=int, default=1e10,
                        help="Number of epoch.")   
    parser.add_argument("--length", type=int, default=1e10,
                        help="length of dataset.")     
    return parser.parse_args()


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers, path, length):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers= extracted_layers
        self.path = path
        self.length = length
        for i in extracted_layers:
            feat_path = osp.join(path, i+'_wh')
            if not os.path.exists(feat_path):
                os.makedirs(feat_path)
            feat_path = osp.join(path, i+'_rc')
            if not os.path.exists(feat_path):
                os.makedirs(feat_path)            
 
    def forward(self, x, image=''):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name in self.extracted_layers:
                #print name
                subsubmodule=module._modules.items()
                y = subsubmodule[0][1](x)
                features = y.data.cpu().numpy()
                num = 2000000 * 512 / features.shape[1] / self.length
                if num > features.shape[2]**2:
                    num = features.shape[2]**2
                features = features.reshape(features.shape[1], -1).transpose(1,0)
                features = random.sample(features, num)
                np.save(osp.join(self.path, name+'_wh', image), features)
  
                y = subsubmodule[1][1](y)
                features = y.data.cpu().numpy()
                features = features.reshape(features.shape[1], -1).transpose(1,0)
                features = random.sample(features, num)       
                np.save(osp.join(self.path, name+'_rc', image), features)

                x = x + y
                x = subsubmodule[2][1](x)
            else:
                x = module(x)
        return x


args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.gpu = 0

model = VGG(num_classes=args.num_classes)
pretrained = True
if pretrained:
    params = torch.load(args.snapshot_dir)['state_dict']
    new_params = model.state_dict().copy()
    for i in params:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if i_parts[0] != 'fc8':
            new_params[i] = params[i]
model.load_state_dict(new_params)

cudnn.benchmark = True

Trainloader = data.DataLoader(
    DataSet(args.data_dir, start=args.start, end=args.end, mean=IMG_MEAN),
    batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

model.cuda(args.gpu)  
model.eval()
exact_list = ['adpater1_1', 'adpater1_2', 'adpater2_1', 'adpater2_2', 'adpater3_1', 'adpater3_2', 'adpater3_3', 
              'adpater4_1', 'adpater4_2', 'adpater4_3', 'adpater5_1', 'adpater5_2', 'adpater5_3', 'adpater6', 'adpater7', ]
path = osp.join(args.data_dir, 'features/')
print args.data_dir.split('/')
extractor = FeatureExtractor(model, exact_list, path, args.length)
for i_iter, data in enumerate(Trainloader):
    images, labels, _, name, = data
    images = Variable(images).type(torch.FloatTensor).cuda(args.gpu)    
    output = extractor(images, name[0].split('/')[-1])
    if i_iter % 100 == 0:
        print i_iter
    


