import scipy.io as sio
import cv2
from CNN_Pa1 import CNN_Pa1
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import math
import random

img = sio.loadmat('PCrot.mat')
img = img['ans']
gt = sio.loadmat('PaviaU_gt.mat')
gt = gt['ans']

num = 0
middata = np.zeros([img.shape[0], img.shape[1]])
midgt = np.zeros([img.shape[0], 1])
for i in range(img.shape[0]):
    if gt[i] != 0:
        middata[num, :] = img[i, :]
        midgt[num] = gt[i]
        num += 1
data = middata[:num]
gt = midgt[:num]

net = CNN_Pa1(5, 9)

'''
data = data.resize(340, 610, 103).detach().numpy()
label = label.resize(340, 610, 1).detach().numpy()
cv2.imshow('data', data[:, :, 0])
cv2.imshow('label', label[:, :, 0])
cv2.waitKey()
'''
net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
lossfc = nn.MSELoss()

loss_ = []


def datachoose(data, gt, pre, num):
    N = np.zeros([num, 1])
    for i in range(data.shape[0]):
        N[int(gt[i]) - 1] += 1
    maxnum = max(N)
    name = np.zeros([num, int(maxnum)])
    midN = np.zeros([num, 1])
    aimnum = np.zeros([num, 1])
    for i in range(num):
        aimnum[i] = math.floor(N[i] * pre)
    sumdata = np.zeros([int(sum(np.array(aimnum))), data.shape[1]])
    sumgt = np.zeros([int(sum(np.array(aimnum))), 1])
    for i in range(data.shape[0]):
        name[int(gt[i]) - 1, int(midN[int(gt[i]) - 1])] = i
        midN[int(gt[i]) - 1] += 1
    middnum = 0
    for i in range(num):
        L = random.sample(range(0, int(N[i])), int(aimnum[i]))
        for j in range(int(aimnum[i])):
            sumdata[middnum, :] = data[int(name[i, int(L[j])]), :]
            sumgt[middnum] = i + 1
            middnum += 1

    newgt = np.zeros([sumgt.shape[0], 9])
    for i in range(sumgt.shape[0]):
        newgt[i, int(sumgt[i] - 1)] = 100
    # print(newgt)
    sumgt = newgt
    return sumdata, sumgt


pre = 0.3
newdata, newlabel = datachoose(data, gt, pre, 9)
newdata = torch.tensor(newdata, dtype=torch.float).resize(newdata.shape[0], 1, 5)
newlabel = torch.tensor(newlabel, dtype=torch.float).resize(newdata.shape[0], 9)
newdata = newdata.cuda()
newlabel = newlabel.cuda()


def train():
    y = net(newdata)
    loss = lossfc(y, newlabel)
    loss_.append(loss.cpu().detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for i in tqdm(range(200)):
    train()

# print(loss_)
plt.figure()
plt.plot(loss_)
plt.figure()
plt.plot(loss_[1000:])
plt.show()

'''def test():
    pre = 0.6
    testdata, testlabel = datachoose(data, gt, pre, 9)
    testdata = torch.tensor(testdata, dtype=torch.float).resize(testdata.shape[0], 1, 5)
    testdata = testdata.cuda()
    y = net(testdata).cpu().detach().numpy()
    # y = np.around(y)
    testnum = 0
    for i in range(testdata.shape[0]):
        print(np.argmax(y[i, :]), np.argmax(testlabel[i, :]))
        if np.argmax(y[i, :]) == np.argmax(testlabel[i, :]):
            testnum += 1
    print(testnum / testdata.shape[0])'''


def test():
    pre = 0.6
    testdata, testlabel = datachoose(data, gt, pre, 9)
    testdata = torch.tensor(testdata, dtype=torch.float).resize(testdata.shape[0], 1, 5)
    testdata = testdata.cuda()
    y = net(testdata).cpu().detach().numpy()
    # y = np.around(y)
    testnum = 0
    testmatrix = np.zeros([9, 9])
    for i in range(testdata.shape[0]):
        testmatrix[np.argmax(y[i, :]), np.argmax(testlabel[i, :])] += 1
        # print(np.argmax(y[i, :]), np.argmax(testlabel[i, :]))
        if np.argmax(y[i, :]) == np.argmax(testlabel[i, :]):
            testnum += 1
    p1 = 0
    p2 = 0
    for i in range(9):
        p1 += testmatrix[i, i]
        p2 += sum(testmatrix[i, :]) * sum(testmatrix[:, i])
    p1 = p1 / testdata.shape[0]
    p2 = p2 / (testdata.shape[0] * testdata.shape[0])
    kappa = (p1 - p2) / (1 - p2)
    print('精度: ', testnum / testdata.shape[0])
    print('Kappa系数: ', kappa)


test()

'''newgt = sio.loadmat('PaviaU_gt.mat')
newgt = newgt['ans']'''
fordata = torch.tensor(img, dtype=torch.float).resize(img.shape[0], 1, 5)
fordata = fordata.cuda()
answer = net(fordata).cpu().detach().numpy()
import creatimg

creatimg.creatimg(answer)
