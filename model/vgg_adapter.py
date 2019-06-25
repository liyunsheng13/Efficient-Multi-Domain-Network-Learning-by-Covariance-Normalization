import numpy as np
import torch
from torch import nn
from torchvision import models


class Adapter(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super(Adapter, self).__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)       
        self.bn2 = nn.BatchNorm2d(dim)


    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        
        out += residual
        out = self.bn2(out)        

        return out


class VGG(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=True):
        super(VGG, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.adpater1_1 = Adapter(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.adpater1_2 = Adapter(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
    
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.adpater2_1 = Adapter(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.adpater2_2 = Adapter(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
    
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.adpater3_1 = Adapter(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.adpater3_2 = Adapter(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.adpater3_3 = Adapter(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
    
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.adpater4_1 = Adapter(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.adpater4_2 = Adapter(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.adpater4_3 = Adapter(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
    
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.adpater5_1 = Adapter(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.adpater5_2 = Adapter(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.adpater5_3 = Adapter(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
    
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.adpater6 = Adapter(4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
    
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.adpater7 = Adapter(4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()  
        
        # fc8
        self.fc8 = nn.Conv2d(4096, num_classes, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.running_var.data.fill_(1)
                m.running_mean.data.zero_()
                    
    def forward(self, x):
        h = x
        h = self.relu1_1(self.adpater1_1(self.conv1_1(h)))
        h = self.relu1_2(self.adpater1_2(self.conv1_2(h)))
        h = self.pool1(h)

        h = self.relu2_1(self.adpater2_1(self.conv2_1(h)))
        h = self.relu2_2(self.adpater2_2(self.conv2_2(h)))
        h = self.pool2(h)

        h = self.relu3_1(self.adpater3_1(self.conv3_1(h)))
        h = self.relu3_2(self.adpater3_2(self.conv3_2(h)))
        h = self.relu3_3(self.adpater3_3(self.conv3_3(h)))
        h = self.pool3(h)
        
        h = self.relu4_1(self.adpater4_1(self.conv4_1(h)))
        h = self.relu4_2(self.adpater4_2(self.conv4_2(h)))
        h = self.relu4_3(self.adpater4_3(self.conv4_3(h)))
        h = self.pool4(h)

        h = self.relu5_1(self.adpater5_1(self.conv5_1(h)))
        h = self.relu5_2(self.adpater5_2(self.conv5_2(h)))
        h = self.relu5_3(self.adpater5_3(self.conv5_3(h)))
        h = self.pool5(h)

        h = self.relu6(self.adpater6(self.fc6(h)))
        h = self.drop6(h)

        h = self.relu7(self.adpater7(self.fc7(h)))
        h = self.drop7(h)

        h = self.fc8(h)
        
        return h 
    
    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size())) 
    def get_parameters(self, fc8 = False):
        b = []
        if fc8 == True:
            b.append(self.fc8)
            '''
            b.append(self.adpater1_1)
            b.append(self.adpater1_2)
            b.append(self.adpater2_1)
            b.append(self.adpater2_2)
            b.append(self.adpater3_1)
            b.append(self.adpater3_2)
            b.append(self.adpater3_3)
            b.append(self.adpater4_1)
            b.append(self.adpater4_2)
            b.append(self.adpater4_3)
            b.append(self.adpater5_1)
            b.append(self.adpater5_2)
            b.append(self.adpater5_3)
            b.append(self.adpater6)
            b.append(self.adpater7) 
            '''
        else:
            b.append(self.adpater1_1)
            b.append(self.adpater1_2)
            b.append(self.adpater2_1)
            b.append(self.adpater2_2)
            b.append(self.adpater3_1)
            b.append(self.adpater3_2)
            b.append(self.adpater3_3)
            b.append(self.adpater4_1)
            b.append(self.adpater4_2)
            b.append(self.adpater4_3)
            b.append(self.adpater5_1)
            b.append(self.adpater5_2)
            b.append(self.adpater5_3)            
            b.append(self.adpater6)
            b.append(self.adpater7)
            '''
            b.append(self.conv1_1)
            b.append(self.conv1_2)
            b.append(self.conv2_1)
            b.append(self.conv2_2)
            b.append(self.conv3_1)
            b.append(self.conv3_2)
            b.append(self.conv3_3)
            b.append(self.conv4_1)
            b.append(self.conv4_2)
            b.append(self.conv4_3)
            b.append(self.conv5_1)
            b.append(self.conv5_2)
            b.append(self.conv5_3)
            b.append(self.fc6)
            b.append(self.fc7)
            '''
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k            