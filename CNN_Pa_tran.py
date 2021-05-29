import scipy.io as sio
import cv2
from CNN_Pa import CNN_Pa
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import math
import random

img = sio.loadmat('PCrot.mat')
img = np.array(img['ans'])
gt = sio.loadmat('PaviaU_gt.mat')
gt = np.array(gt['ans'])

data = np.resize(img, [340, 610, 5])
label = np.resize(gt, [340, 610, 1])
data = np.pad(data, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=0)
print(data.shape)
'''labelt=np.zeros([340,610,1])
for i in range(340):
    for j in range(610):
        labelt[i,j]=gt[i*610+j]
labelt=np.array(labelt)
print(labelt.shape)

cv2.imshow('data', data[:, :, 0:3])
#cv2.imshow('label', labelt)
cv2.waitKey()
'''
data2d = []
for i in range(340):
    for j in range(610):
        # tem=np.zeros([9,9,5])
        tem = data[i:i + 9, j:j + 9, :]
        data2d.append(tem)
data2d = np.array(data2d)
print(data2d.shape)

num = 0
middata = np.zeros([data2d.shape[0], 9, 9, 5])
midgt = np.zeros([data2d.shape[0], 1])
for i in range(data2d.shape[0]):
    if gt[i] != 0:
        middata[num, :, :, :] = data2d[i, :, :, :]
        midgt[num] = gt[i]
        num += 1
data = middata[:num]
gt = midgt[:num]

net = CNN_Pa(9, 9)

net.cuda()
############################################################################################################
net_name = 'CNN_Pa'
lr_tran = 1
pre_tran = 0.3
num_tran = 2000
loss_fname = 'loss/' + str(net_name) + '_loss_' + str(lr_tran) + '_' + str(pre_tran) + '_' + str(num_tran) + '.mat'
log_fname = 'log/' + str(net_name) + '_log_' + str(lr_tran) + '_' + str(pre_tran) + '_' + str(num_tran) + '.txt'
img_fname = 'img/' + str(net_name) + '_img_' + str(lr_tran) + '_' + str(pre_tran) + '_' + str(num_tran) + '.jpg'
############################################################################################################

optimizer = torch.optim.Adam(net.parameters(), lr=lr_tran)
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
    sumdata = np.zeros([int(sum(np.array(aimnum))), 9, 9, 5])
    sumgt = np.zeros([int(sum(np.array(aimnum))), 1])
    for i in range(data.shape[0]):
        name[int(gt[i]) - 1, int(midN[int(gt[i]) - 1])] = i
        midN[int(gt[i]) - 1] += 1
    middnum = 0
    for i in range(num):
        L = random.sample(range(0, int(N[i])), int(aimnum[i]))
        for j in range(int(aimnum[i])):
            sumdata[middnum, :, :, :] = data[int(name[i, int(L[j])]), :, :, :]
            sumgt[middnum] = i + 1
            middnum += 1

    newgt = np.zeros([sumgt.shape[0], 9])
    for i in range(sumgt.shape[0]):
        newgt[i, int(sumgt[i] - 1)] = 100
    # print(newgt)
    sumgt = newgt
    return sumdata, sumgt


newdata, newlabel = datachoose(data, gt, pre_tran, 9)
newdata = torch.tensor(newdata, dtype=torch.float).resize(newdata.shape[0], 5, 9, 9)
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
    testdata = torch.tensor(testdata, dtype=torch.float).resize(testdata.shape[0], 5, 9, 9)
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
        f.write('\nCNN_Pa_tran\n训练次数：' + str(num_tran) + '    样本比例：' + str(pre_tran)
                + '    学习率：' + str(lr_tran) + '    优化器：Adam    损失函数：MSELoss')
        f.write('\n精度: ' + str(testnum / testdata.shape[0]))
        f.write('\nKappa系数: ' + str(kappa))
        f.write('\n混淆矩阵： ' + str(testmatrix.tolist()))
        f.write('\n每类正确分类概率：' + str(crop))
    print('精度: ', testnum / testdata.shape[0])
    print('Kappa系数: ', kappa)


test()
answers = []
for i in range(100):
    datatem = data2d[int(data2d.shape[0] / 100 * i):int(data2d.shape[0] / 100 * (i + 1))]
    # print(datatem.shape)
    data1 = torch.tensor(datatem, dtype=torch.float).resize(datatem.shape[0], 5, 9, 9).cuda()
    answer1 = net(data1).cpu().detach().numpy()
    # print(i)
    if i == 0:
        answers = answer1
    else:
        answers = np.vstack([answers, answer1])
# print(answers.shape)

import creatimg

creatimg.creatimg(answers, img_fname)
