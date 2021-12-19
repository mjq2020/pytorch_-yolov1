#2021/11/13 11:16
import torch
import torchvision
import os
import torch.nn as nn
from torch.nn.modules import Sequential


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),padding=3,stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),padding=3,stride=1),
            nn.MaxPool2d(kernel_size=(2,2),stride=2),
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2,2),stride=2),
            nn.Conv2d(in_channels=192,out_channels=128,kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #四个卷积
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            #
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #连续四层卷积
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            #最后四层卷积
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
            nn.Linear(1024*7*7,1024*7),
            nn.BatchNorm1d(1024*7),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(),
            nn.Linear(1024 * 7, 4096),
            # nn.BatchNorm1d(4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(),
            nn.Linear(4096,7*7*30),
            nn.ReLU()
            )


    def forward(self,x):
        print(x.shape)
        x=self.seq(x)
        print(x.shape)
        x=x.reshape(-1,7,7,30)
        # print(x.shape)
        return x

if __name__ == '__main__':
    model=Net()
    print(model)