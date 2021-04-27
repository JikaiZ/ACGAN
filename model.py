import time

import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

class Generator_128(nn.Module):
    def __init__(self):
        super(Generator_128, self).__init__()
        self.fc1 = nn.Linear(110, 384)
        
        self.transConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=4, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.transConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )
        self.transConv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
        self.transConv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )
        self.transConv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(True)
        )
        self.transConv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2,padding=1,bias=False),
            nn.Tanh()
        )
#         self.apply(weights_init)
        
    def forward(self, x):
        x = x.view(-1, 110)
        fc1 = self.fc1(x)
        fc1 = fc1.view(-1, 384, 1, 1)
        transconv1 = self.transConv1(fc1)
        transconv2 = self.transConv2(transconv1)
        transconv3 = self.transConv3(transconv2)
        transconv4 = self.transConv4(transconv3)
        transconv5 = self.transConv5(transconv4)
        transconv6 = self.transConv6(transconv5)
        return transconv6
    
class Generator_64(nn.Module):
    def __init__(self):
        super(Generator_64, self).__init__()
        self.fc1 = nn.Linear(110, 384)
        
        self.transConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=4, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.transConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )
        self.transConv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
        self.transConv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )
        self.transConv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=24, out_channels=3, kernel_size=4, stride=2,padding=1,bias=False),
            nn.Tanh()
        )
#         self.transConv6 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2,padding=1,bias=False),
#             nn.Tanh()
#         )
#         self.apply(weights_init)
        
    def forward(self, x):
        x = x.view(-1, 110)
        fc1 = self.fc1(x)
        fc1 = fc1.view(-1, 384, 1, 1)
        transconv1 = self.transConv1(fc1)
        transconv2 = self.transConv2(transconv1)
        transconv3 = self.transConv3(transconv2)
        transconv4 = self.transConv4(transconv3)
        transconv5 = self.transConv5(transconv4)
        return transconv5

class Generator_32(nn.Module):
    def __init__(self):
        super(Generator_32, self).__init__()
        self.fc1 = nn.Linear(110, 384)
        
        self.transConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=4, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.transConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )
        self.transConv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2,padding=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )
        self.transConv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=3, kernel_size=4, stride=2,padding=1,bias=False),
            nn.Tanh()
        )
#         self.transConv5 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=24, out_channels=3, kernel_size=4, stride=2,padding=1,bias=False),
#             nn.Tanh()
#         )
#         self.transConv6 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2,padding=1,bias=False),
#             nn.Tanh()
#         )
#         self.apply(weights_init)
        
    def forward(self, x):
        x = x.view(-1, 110)
        fc1 = self.fc1(x)
        fc1 = fc1.view(-1, 384, 1, 1)
        transconv1 = self.transConv1(fc1)
        transconv2 = self.transConv2(transconv1)
        transconv3 = self.transConv3(transconv2)
        transconv4 = self.transConv4(transconv3)
        return transconv4

class Discriminator(nn.Module):
    def __init__(self, input_size, num_labels):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.fc_aux = nn.Linear(in_features=512*int(input_size / 8)*int(input_size / 8), out_features=num_labels)
        self.fc = nn.Linear(512*int(input_size / 8)*int(input_size / 8), 1)
        self._softmax = nn.Softmax()
        self._sigmoid = nn.Sigmoid()
#         self.apply(weights_init)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)
        x = x.view(x.size(0), -1)
        aux = self._softmax(self.fc_aux(x))
        fc = self._sigmoid(self.fc(x)).view(-1, 1).squeeze(1)
        return fc, aux
        
class ResNet_Block(nn.Module):
    def __init__(self, in_chs, out_chs, strides):
        super(ResNet_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                      stride=strides, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_chs, out_channels=out_chs,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_chs))

        if in_chs != out_chs:
            self.id_mapping = nn.Sequential(
                nn.Conv2d(in_channels=in_chs, out_channels=out_chs,
                          stride=strides, padding=0, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chs))
        else:
            self.id_mapping = None
        self.final_activation = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.id_mapping is not None:
            x_ = self.id_mapping(x)
        else:
            x_ = x
        return self.final_activation(x_ + out)

class ResNetCIFAR(nn.Module):
    def __init__(self, num_layers=20, num_stem_conv=32, config=(16, 32, 64)):
        super(ResNetCIFAR, self).__init__()
        self.num_layers = 20
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_stem_conv,
                      stride=1, padding=1, kernel_size=3, bias=False),
            nn.BatchNorm2d(num_stem_conv),
            nn.ReLU(True)
        )
        num_layers_per_stage = (num_layers - 2) // 6
        self.body_op = []
        num_inputs = num_stem_conv
        for i in range(len(config)):
            for j in range(num_layers_per_stage):
                if j == 0 and i != 0:
                    strides = 2
                else:
                    strides = 1
                self.body_op.append(ResNet_Block(num_inputs, config[i], strides))
                num_inputs = config[i]
        self.body_op = nn.Sequential(*self.body_op)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.final_fc = nn.Linear(config[-1], 10)

    def forward(self, x):
        out = self.head_conv(x)
        out = self.body_op(out)
        self.features = self.avg_pool(out)
        self.feat_1d = self.features.mean(3).mean(2)
        return self.final_fc(self.feat_1d)