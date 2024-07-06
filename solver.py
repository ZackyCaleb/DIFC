import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Networks import Feature, Predictor
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import albumentations
import argparse
from tqdm import tqdm
import os
from Dataset import RAF_Paths, Target_train_Paths, Target_test_Paths


class Solver(object):
    def __init__(self, num_k=4, batch_size=32, lr=0.001, momentum=0.9, size=224, mode='sfew',
                 sour_path=None, target_train_path=None, target_test_path=None):
        super(Solver, self).__init__()
        self.batch_size = batch_size
        self.num_k = num_k
        self.lr = lr
        self.G = Feature()
        self.C1 = Predictor()
        self.C2 = Predictor()

        train_sour_data = RAF_Paths(sour_path, size)
        self.train_sour_loader = DataLoader(train_sour_data, batch_size=batch_size, shuffle=True)
        train_target_data = Target_train_Paths(target_train_path, size)
        self.train_target_loader = DataLoader(train_target_data, batch_size=batch_size, shuffle=True)
        test_target_data = Target_test_Paths(target_test_path, mode, size)
        self.test_target_loader = DataLoader(test_target_data, batch_size=batch_size, shuffle=False)

        self.opt_g = optim.Adam(self.G.parameters(), lr=self.lr, weight_decay=0.0005)

        self.opt_c1 = optim.Adam(self.C1.parameters(), lr=self.lr, weight_decay=0.0005)
        self.opt_c2 = optim.Adam(self.C2.parameters(), lr=self.lr, weight_decay=0.0005)

    def init_weight1(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    def init_weight2(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight, gain=0.1)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

        self.C1.apply(self.init_weight1)
        self.C2.apply(self.init_weight2)


        # self.opt_g = optim.SGD(self.G.parameters(),
        #                        lr=lr, weight_decay=0.0005,
        #                        momentum=momentum)
        #
        # self.opt_c1 = optim.SGD(self.C1.parameters(),
        #                         lr=lr, weight_decay=0.0005,
        #                         momentum=momentum)
        # self.opt_c2 = optim.SGD(self.C2.parameters(),
        #                         lr=lr, weight_decay=0.0005,
        #                         momentum=momentum)

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

    def ce_loss(self, input, target):
        logit = nn.functional.log_softmax(input, dim=1)
        loss = -torch.sum(logit * target) / (input.shape[0])
        return loss

    def train(self, epoch):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch1, batch2 in zip(self.train_sour_loader, self.train_target_loader):
            img_s, label_s = batch1[0], batch1[1]
            img_t = batch2
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)

            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)

            output_s2 = self.C2(feat_s)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            # loss_s1 = self.ce_loss(output_s1, label_s)
            # loss_s2 = self.ce_loss(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            # loss_s1 = self.ce_loss(output_s1, label_s)
            # loss_s2 = self.ce_loss(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in range(self.num_k):
                #
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            print('Train Epoch: {} [ Loss1:{:.6f} \t Loss2:{:.6f} \t Discrepency: {:.6f}]'.format(
                epoch, loss_s1.cpu().detach().numpy(), loss_s2.cpu().detach().numpy(), loss_dis.cpu().detach().numpy()))

    def test(self, epoch):
        nll_loss = nn.CrossEntropyLoss().cuda()
        with torch.no_grad():
            self.G.eval()
            self.C1.eval()
            self.C2.eval()
            test_loss = 0
            correct1 = 0
            correct2 = 0
            correct3 = 0
            size = len(self.test_target_loader.sampler)
            loop = tqdm(self.test_target_loader, desc='test')
            for (images, labels) in loop:
                img, label = images.cuda(), labels.cuda()
                # img, label = Variable(img, volatile=True), Variable(label)
                feat = self.G(img)
                output1 = self.C1(feat)
                output2 = self.C2(feat)

                test_loss += nll_loss(output1, label).data
                output_ensemble = output1 + output2
                pred1 = output1.data.max(1)[1]
                pred2 = output2.data.max(1)[1]
                pred_ensemble = output_ensemble.data.max(1)[1]

                correct1 += torch.sum(pred1.data == torch.argmax(label.data, dim=1))
                correct2 += torch.sum(pred2.data == torch.argmax(label.data, dim=1))
                correct3 += torch.sum(pred_ensemble.data == torch.argmax(label.data, dim=1))
                # correct1 += pred1.eq(label.data).cpu().sum()
                # correct2 += pred2.eq(label.data).cpu().sum()
                # correct3 += pred_ensemble.eq(label.data).cpu().sum()

            test_loss = test_loss / size
            print(
                '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                    test_loss, correct1, size,
                    100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size,
                    100. * correct3 / size))
