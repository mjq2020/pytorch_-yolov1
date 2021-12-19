#2021/11/16 13:20

import torch
import numpy as np
import torch
from tqdm.std import tqdm

import torchvision
from torch import optim
from utils.data_loader import Dataer
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from utils.yolo_loss import Loss
from model.yolo import Net
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# @staticmethod


def collate_fn(batch):
    img, label = zip(*batch)  # transposed
    # for i, l in enumerate(label):
    #     l[:, 0] = 1  # add target image index for build_targets()
    return torch.stack(img, 0), label


def print_params(params):
    for name, parms in params:
        # print('-->name:', name)
        # print('-->para:', parms)
        print('-->grad_requirs:', parms.requires_grad)
        print('-->grad_value:', parms.grad)
        break


def re_model():
    model = Net()
    model.to(device)
    # model.load_state_dict(torch.load('./weights/best.pt'))
    return model


#加载模型
model=re_model()

#加载自定义损失函数及优化器
loss=Loss()
opt=optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0005)

#加载自定义数据集
trainset=Dataer(r'.\data\train.txt')
traindata=DataLoader(trainset,batch_size=4,shuffle=True,drop_last=True)

#加载日志模块
writer=SummaryWriter('./log/train')

step=1
init_loss=1000
epcoh=20
for epoch in range(150):
    print(f'=========第{epoch+1}轮==========')
    runing_loss=0
    ind = 1
    model.train()

    for data in tqdm(traindata):
        img,labels=data
        img,labels=img.to(device),labels.to(device)
        opt.zero_grad()
        output=model(img)
        ll=loss(output,labels)
        print(ll.item())
        ll.backward()
        # print_params(model.named_parameters())
        opt.step()
        lo=ll.item()
        del ll,output           #防止内存累积
        runing_loss+=lo
        if ind%epcoh==0:
            avgloss=runing_loss/epcoh
            print(f'loss：{avgloss}')
            writer.add_scalar('LOSS3',avgloss,step)
            step+=1
            if avgloss<init_loss:
                torch.save(model.state_dict(),'./weights/best.pt')
                init_loss=avgloss
            runing_loss = 0

        ind+=1