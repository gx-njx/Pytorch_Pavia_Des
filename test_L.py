import scipy.io as sio
import cv2
from Linear_Pa import Linear_Pa
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

net = Linear_Pa(img.shape[1], 9)

'''
data = data.resize(340, 610, 103).detach().numpy()
label = label.resize(340, 610, 1).detach().numpy()
cv2.imshow('data', data[:, :, 0])
cv2.imshow('label', label[:, :, 0])
cv2.waitKey()
'''
net.cuda()
############################################################################################################
net_name = 'Linear_Pa'
lr_tran = 0.1
pre_tran = 0.3
num_tran = 2000
loss_fname = 'loss/' + str(net_name) + '_loss_' + str(lr_tran) + '_' + str(pre_tran) + '_' + str(num_tran) + '.mat'
log_fname = 'log/' + str(net_name) + '_log_' + str(lr_tran) + '_' + str(pre_tran) + '_' + str(num_tran) + '.txt'
img_fname = 'img/' + str(net_name) + '_img_' + str(lr_tran) + '_' + str(pre_tran) + '_' + str(num_tran) + '.jpg'
############################################################################################################

optimizer = torch.optim.Adam(net.parameters(), lr=lr_tran)
lossfc = nn.MSELoss()

loss_ = []

'''
def datachoose(data, gt, pre, num):
    num = data.shape[0] * pre
    L = random.sample(range(0, data.shape[0] - 1), num)
    newdata = np.zeros([num, data.shape[1]])
    newgt = np.zeros([num, 9])
    for i in range(num):
        newdata[i, :] = data[L[i], :]
        newgt[i, :] = gt[L[i], :]
    return newdata, newgt
'''


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


pre = pre_tran
newdata, newlabel = datachoose(data, gt, pre, 9)
count = newdata.shape[0]
# print(newdata,newlabel)
newdata = torch.tensor(newdata, dtype=torch.float).resize(count, 5)
newlabel = torch.tensor(newlabel, dtype=torch.float).resize(count, 9)
newdata = newdata.cuda()
newlabel = newlabel.cuda()


def train():
    y = net(newdata)
    loss = lossfc(y, newlabel)
    loss_.append(loss.cpu().detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for i in tqdm(range(num_tran)):
    train()

# print(loss_)
sio.savemat(loss_fname, {'loss': np.array(loss_)})
'''plt.figure()
plt.plot(loss_)
plt.figure()
plt.plot(loss_[1000:])
plt.show()'''
'''torch.save(net,'CNN_Pa_net.pkl')
net=torch.load('CNN_Pa_net.pkl')'''


def test():
    pre = 0.6
    testdata, testlabel = datachoose(data, gt, pre, 9)
    count = testdata.shape[0]
    testdata = torch.tensor(testdata, dtype=torch.float).resize(count, 5)
    testdata = testdata.cuda()
    y = net(testdata).cpu().detach().numpy()
    # y = np.max(y)
    testnum = 0
    testmatrix = np.zeros([9, 9])
    for i in range(count):
        testmatrix[np.argmax(y[i, :]), np.argmax(testlabel[i, :])] += 1
        # print(np.argmax(y[i, :]), np.argmax(testlabel[i, :]))
        if np.argmax(y[i, :]) == np.argmax(testlabel[i, :]):
            testnum += 1
    p1 = 0
    p2 = 0
    crop = []
    for i in range(9):
        p1 += testmatrix[i, i]
        p2 += sum(testmatrix[i, :]) * sum(testmatrix[:, i])
        crop.append(testmatrix[i, i] / sum(testmatrix[:, i]))
    p1 = p1 / testdata.shape[0]
    p2 = p2 / (testdata.shape[0] * testdata.shape[0])
    kappa = (p1 - p2) / (1 - p2)
    with open(log_fname, 'w') as f:
        import time
        f.write('\n' + time.strftime('%Y/%m/%d--%H:%M:%S') + '-----' * 10)
        f.write('\nLinear_Pa\n训练次数：' + str(num_tran) + '    样本比例：' + str(pre_tran)
                + '    学习率：' + str(lr_tran) + '    优化器：Adam    损失函数：MSELoss')
        f.write('\n精度: ' + str(testnum / testdata.shape[0]))
        f.write('\nKappa系数: ' + str(kappa))
        f.write('\n混淆矩阵： ' + str(testmatrix.tolist()))
        f.write('\n每类正确分类概率：' + str(crop))
    print('精度: ', testnum / testdata.shape[0])
    print('Kappa系数: ', kappa)


test()

fordata = torch.tensor(img, dtype=torch.float).resize(img.shape[0], 5)
fordata = fordata.cuda()
answer = net(fordata).cpu().detach().numpy()
import creatimg

creatimg.creatimg(answer, img_fname)
