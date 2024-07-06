import torch
import torch.nn as nn
from ResNet import resnet_v1_18,resnet50
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from recordermeter import RecorderMeter


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        exp_resnet = resnet_v1_18(pretrained=False)
        cele_check = torch.load(r'F:\FER_dataset_clearned\Pre-trained model\resnet18_msceleb.pth')
        exp_resnet.load_state_dict(cele_check['state_dict'], strict=True)
        self.encoder1 = nn.Sequential(*list(exp_resnet.children())[:-2])

        # resnet = resnet50()
        # resnet.fc = nn.Linear(2048, 12666)
        # checkpoint = torch.load(r'F:\FER_dataset_clearned\Pre-trained model\resnet50_pretrained_on_msceleb.pth.tar')
        # pre_trained_dict = checkpoint['state_dict']
        # model_dict = resnet.state_dict()
        # # for k, v in pre_trained_dict.items():
        # #     if k in model_dict:
        # #         print(k, v.shape)
        # pretrained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # resnet.load_state_dict(model_dict)
        # self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.encoder1(x)
        f = self.avg(f)
        f = f.view(f.shape[0], -1)
        return f

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

# class Predictor(nn.Module):
#     def __init__(self, prob=0.5):
#         super(Predictor, self).__init__()
#         self.fc1 = nn.Linear(512, 256)
#         self.bn1_fc = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 128)
#         self.bn2_fc = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, 7)
#         self.bn_fc3 = nn.BatchNorm1d(7)
#         self.prob = prob
#
#     def set_lambda(self, lambd):
#         self.lambd = lambd
#
#     def forward(self, x, reverse=False):
#         if reverse:
#             x = grad_reverse(x, self.lambd)
#         x = F.relu(self.bn1_fc(self.fc1(x)))
#         x = F.relu(self.bn2_fc(self.fc2(x)))
#         x = self.fc3(x)
#         return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2_fc = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 7)
        self.bn_fc3 = nn.BatchNorm1d(7)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        # x = self.fc1(x)
        return x

# class Feature(nn.Module):
#     def __init__(self):
#         super(Feature, self).__init__()
#         resnet = resnet50()
#         resnet.fc = nn.Linear(2048, 12666)
#         checkpoint = torch.load(r'F:\FER_dataset_clearned\Pre-trained model\resnet50_pretrained_on_msceleb.pth.tar')
#         pre_trained_dict = checkpoint['state_dict']
#         model_dict = resnet.state_dict()
#         # for k, v in pre_trained_dict.items():
#         #     if k in model_dict:
#         #         print(k, v.shape)
#         pretrained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
#         model_dict.update(pretrained_dict)
#         resnet.load_state_dict(model_dict)
#         self.encoder1 = nn.Sequential(*list(resnet.children())[:-3])
#         self.encoder2 = nn.Sequential(*list(resnet.children())[:-5])
#
#         self.conv1 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)
#         self.avg1 = nn.AdaptiveAvgPool2d(1)
#
#
#     def forward(self, x):
#         f = self.encoder(x)
#         f1 = self.conv1(f)
#         f1 = self.avg1(f1)
#         f1 = f.view(f.shape[0], -1)