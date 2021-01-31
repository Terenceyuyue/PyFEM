# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:39:40 2021

@author: Terenceyuyue
"""

## Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

## Parameters
dim_input, dim_hidden, dim_output = 28*28, 15, 10
epochs = 10
mini_batch_size = 10
eta = 1e-3

## Load MNIST
# 下载图片和标签
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())
# 数据加载
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=mini_batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=mini_batch_size, 
                                          shuffle=False)

## Define the network
class network(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(network, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_hidden) 
        self.fc2 = nn.Linear(dim_hidden, dim_output)
    
    def forward(self, x):
        y = self.fc1(x)
        y = torch.sigmoid(y) # torch.relu(y)
        y = self.fc2(y)
        return y

## Create a Network object
model = network(dim_input, dim_hidden, dim_output)


## Train the network
# loss function
criterion = nn.CrossEntropyLoss() 
# optimzer
optimizer = torch.optim.Adam(model.parameters(), lr=eta)  
# loop of epochs
for ep in range(epochs):
    for images, labels in train_loader:   

        # feedforward
        images = images.reshape(-1, 28*28) # 有可能后面部分不是 10 个
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()  # compute the gradients
        
        ## train network with given optimizer
        optimizer.step() # update the parameters
        
    
    with torch.no_grad(): # 不构建计算图
        n_correct = 0
        n_test = len(test_dataset)
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            pred = torch.argmax(outputs.data, 1)     
            n_correct += (pred == labels).sum().item()         
        print("Epoch {:2d} : {} / {}".format(ep+1,n_correct,n_test))



