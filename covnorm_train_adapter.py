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
from model.vgg_adapter import VGG
from dataset.dataset import DataSet
from dataset.dataset import TestDataSet
import shutil

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    
    parser.add_argument("--epoch", type=int, default=40,
                        help="Number of epoch.")    
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default='/path/to/dataset',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--snapshot-dir", type=str, default='/path/to/checkpoints',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=2e-3,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gamma", type=int, default=5,
                        help="epochs that decay lr.")    
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="pretrained model for residual adapter.")    
    return parser.parse_args()

def adjust_learning_rate(args, optimizer, power):
    #lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = args.learning_rate * (0.1**power)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = args.learning_rate * (0.1**power)
        
def save_checkpoint(state, is_best, filename='checkpoint.pth', snapshot=''):
    torch.save(state, osp.join(snapshot, filename))
    if is_best:
        shutil.copyfile(osp.join(snapshot, filename), osp.join(snapshot, 'model_best.pth'))

args = get_arguments()

print args

snapshot_dir = osp.join(args.snapshot_dir, args.data_dir.split('/')[-1] + '_lr_%f_wd_%f_ra' %(args.learning_rate, args.weight_decay))
if not os.path.exists(snapshot_dir):
    os.mkdir(snapshot_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.gpu = 0

model = VGG(num_classes=args.num_classes)
vgg16 = models.vgg16()

if args.pretrained is not None:
    #params = torch.load('/data5/yunsheng/MTWCL-pytorch/checkpoints/model_best.pth.tar')['state_dict']
    params = torch.load(args.pretrained)['state_dict']
    new_params = model.state_dict().copy()
    for i in params:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        # print i_parts
        if i_parts[0] != 'fc8':
            new_params[i] = params[i]
model.load_state_dict(new_params)
    


cudnn.benchmark = True

trainloader = data.DataLoader(
    DataSet(args.data_dir, mean=IMG_MEAN),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

trainloader_iter = enumerate(trainloader)

testloader = data.DataLoader(
    TestDataSet(args.data_dir, mean=IMG_MEAN),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

testloader_iter = enumerate(testloader)

optimizer = torch.optim.Adam(
    [
        {'params': model.get_parameters(fc8=False)},
        {'params': model.get_parameters(fc8=True),
         'lr': args.learning_rate},
    ],
    lr=args.learning_rate,
    betas=(0.9, 0.99), weight_decay=args.weight_decay)

softmax_loss = torch.nn.CrossEntropyLoss()
model.cuda(args.gpu)  
best_res, acc = 0, []
for i in range(args.epoch): 
    model.train() 
    for i_iter, data in enumerate(trainloader):
        optimizer.zero_grad()
        images, labels, _, _, = data
        images = Variable(images).type(torch.FloatTensor).cuda(args.gpu)
        labels = Variable(labels.reshape(labels.shape[0], 1, 1)).type(torch.LongTensor).cuda(args.gpu)
        preds = model(images)
        loss = softmax_loss(preds, labels)
        loss.backward()
        optimizer.step()
        #print loss.data.cpu().numpy()

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
    if res > best_res:
        best_res = res
        is_best = True
    else:
        is_best = False
    
    save_checkpoint({
        'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_prec1': res,
                'args': args,
                'optimizer' : optimizer.state_dict(),
                }, is_best, filename='checkpoint.pth', snapshot=snapshot_dir)    
    
    print('Accuracy of the network on the test images: %.4f%%' % (
        100 * correct / total)) 
    acc.append(100 * correct / total)
    with open(osp.join(snapshot_dir, 'log.csv'), 'a') as f:
        log = [i+1, 100 * correct / total]
        log = map(str, log)
        f.write(','.join(log) + '\n')    
        
        
    if (i+1) % args.gamma == 0:
        adjust_learning_rate(args, optimizer, int((i+1)/args.gamma))
print acc
