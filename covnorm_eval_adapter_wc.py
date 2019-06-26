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
import random
from model.vgg_adapter_pca import VGG
from dataset.dataset import DataSet
from dataset.dataset import TestDataSet
from pca.pca_dim_compute import PCA_Dim_Compute
import shutil

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
      
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default='/path/to/dataset',
                        help="Path to the directory containing the source dataset.") 
    parser.add_argument("--pca-ratio", type=float, default=0.995,
                        help="pca component ratio")    
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--snapshot-dir", type=str, default='/path/to/checkpoints',
                        help="Where to save snapshots of the model.")   
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

print args

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.gpu = 0

#pca_dim = np.load(osp.join(args.pca_dir, 'pca_dim.npy'))
#pca_dir = np.load(osp.join(args.pca_dir, 'pca_dir.npy')).item()

pca_dim, pca_dir = PCA_Dim_Compute(os.path.join(args.data_dir, 'features'), args.pca_ratio)
model = VGG(pca_dir=pca_dir, num_classes=args.num_classes, dim=pca_dim, pca=True)
model.load_state_dict(torch.load(args.snapshot_dir)['state_dict'])   

cudnn.benchmark = True

testloader = data.DataLoader(
    TestDataSet(args.data_dir, mean=IMG_MEAN),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

testloader_iter = enumerate(testloader)

softmax_loss = torch.nn.CrossEntropyLoss()
model.cuda(args.gpu)  

total, correct = 0.0, 0.0
model.eval()
for _, data in enumerate(testloader):
    images, labels, _, _, = data
    images = Variable(images).type(torch.FloatTensor).cuda(args.gpu)
    labels = Variable(labels.reshape(labels.shape[0], 1, 1)).type(torch.LongTensor).cuda(args.gpu)        
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()   

res = correct / total    

print('Accuracy of the network on the test images: %.4f%%' % (
    100 * correct / total)) 