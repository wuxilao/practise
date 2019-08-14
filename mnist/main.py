#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from model import LeNet
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100
LR = 0.01
Momentum = 0.9

# 下载数据集
train_dataset = datasets.MNIST(root = './',
                              train=True,
                              transform = transforms.ToTensor(),
                              download=True)
test_dataset =datasets.MNIST(root = './',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
#建立一个数据迭代器
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                         batch_size = batch_size,
                                         shuffle = False)

#实现单张图片可视化
# images,labels = next(iter(train_loader))
# img  = torchvision.utils.make_grid(images)
# img = img.numpy().transpose(1,2,0)
# # img.shape
# std = [0.5,0.5,0.5]
# mean = [0.5,0.5,0.5]
# img = img*std +mean
# cv2.imshow('win',img)
# key_pressed = cv2.waitKey(0)

net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()#定义损失函数
optimizer = torch.optim.SGD(net.parameters(),lr=LR,momentum=Momentum)

epoch = 10
if __name__ == '__main__':
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()#将梯度归零
            outputs = net(inputs)#将数据传入网络进行前向运算
            loss = criterion(outputs, labels)#得到损失函数
            loss.backward()#反向传播
            optimizer.step()#通过梯度做一步参数更新

            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

    #验证测试集
    net.eval()#将模型变换为测试模式
    correct = 0
    total = 0
    for data_test in test_loader:
        images, labels = data_test
        images, labels = images.cuda(), labels.cuda()
        output_test = net(images)
        # print("output_test:",output_test.shape)

        _, predicted = torch.max(output_test, 1)#此处的predicted获取的是最大值的下标
        # print("predicted:",predicted.shape)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct1: ",correct)
    print("Test acc: {0}".format(correct.item() / len(test_dataset)))#.cpu().numpy()
