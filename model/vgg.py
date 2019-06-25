import numpy as np
import torch
from torch import nn
from torchvision import models
class VGG(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=True):
        super(VGG, self).__init__()
        vgg = models.vgg16()
                    

        classifier = list(vgg.classifier.children())

        
        classifier = nn.Sequential(*(classifier[i] for i in range(6)))

        #fc8 = [nn.Linear(4096, num_classes)] 
        self.fc8 = nn.Linear(in_features=4096, out_features=num_classes)

        self.features = vgg.features
        self._initialize_weights()
        
        self.classifier = nn.Sequential(*([classifier[i] for i in range(len(
            classifier))]))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.running_var.data.fill_(1)
                m.running_mean.data.zero_()    


    def forward(self, x):
        x = self.features(x)
        (_, C, H, W) = x.data.size()
        x = x.view( -1 , C * H * W)        
        x = self.classifier(x)
        x = self.fc8(x)
        return x

    def optim_parameters(self, args):
        return self.parameters()
    
    def get_parameters(self, fc8 = False):
        b = []
        if fc8 == True:
            b.append(self.fc8)
        else:
            b.append(self.features)
            b.append(self.classifier)
        
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k        