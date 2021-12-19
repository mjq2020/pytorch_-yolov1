#2021/11/24 10:55
import torchvision
import torch
import torch.nn as nn
from utils.data_loader import Dataer
from torch.utils.data import DataLoader,Dataset
from utils.yolo_loss2 import Loss
import torch.optim as opt
from tqdm.std import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


class Yolo():
    def __init__(self):

        self.epochs=150
        self.batchsize=64
        self.is_bing=True
        self.device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        model=self.load_model()
        trainset=self.load_dataset()
        mse_loss=Loss()
        optim=opt.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0005)
        model.to(self.device)
        self.writer=self.init_write_log()


        prin_loss=1
        img_len=len(trainset)
        step=1
        init_loss=1000
        for epoch in range(self.epochs):
            total_loss=0
            ind = 1
            self.update_lr(optim,epoch)
            lr=self.get_lr(optim)
            for img,labels in tqdm(trainset):

                img,labels=img.to(self.device),labels.to(self.device)
                model.train()
                pre=model(img)
                optim.zero_grad()
                loss,all_loss=mse_loss(pre,labels)
                loss.backward()

                total_loss+=loss.item()
                # self.look_grad(model.named_parameters())
                optim.step()

                if ind%prin_loss==0:
                    avg_loss=total_loss/prin_loss
                    print(f'Epoch:{epoch},Lr:{lr},Images:{ind}/{img_len},AvgLoss:{avg_loss}')
                    self.writer_log(all_loss,step)
                    step+=1
                    if avg_loss<init_loss:
                        torch.save(model.state_dict(),'./weights/best11.pt')
                        init_loss=avg_loss
                    total_loss=0

                ind+=1


    def writer_log(self,data:dict,step):
        self.writer.add_scalars('LOSS',data,step)
        # for key in data.keys():
        #     self.writer.add_scalar(key,data[key],step)


    def load_model(self):

        model=torchvision.models.resnet152(pretrained=False)
        model.to(self.device)
        model.load_state_dict(torch.load('/home/sdjsj/.cache/torch/hub/checkpoints/resnet152-b121ed2d.pth'),)

        self.feature_bing(model)
        model.layer4.add_module('conver1',nn.Sequential(            nn.Conv2d(2048,1024,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),))

        model.fc=nn.Sequential(
            nn.Linear(in_features=1024,out_features=2048,bias=True),
            # nn.BatchNorm1d(1),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.Linear(in_features=2048, out_features=1470, bias=True),
            nn.ReLU()


        )
        # model.fc.requires_grad=True
        # print(model.requires_grad_())
        # exit()
        return model

    def look_grad(self,params):
        for name, parms in params:
            # print('-->name:', name)
            # print('-->para:', parms)
            if parms.requires_grad:
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:', parms.requires_grad)
                print('-->grad_value:', parms.grad.shape)
            # break

    def update_lr(self,optimizer, epoch):
        if epoch >105:
            lr = 0.0001
        elif epoch > 75:
            lr=0.01
        elif epoch > 30:
            lr = 0.005
        elif epoch >= 0:
            lr = 0.001
        else:
            lr=0.005


        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def feature_bing(self,model):
        '''冻结特征提取层'''
        if self.is_bing:
            for param in model.named_parameters():
                # print(param[0])
                if 'fc' in param[0]:continue
                else:
                    # print(param[1].requires_grad)
                    param[1].requires_grad=False
                # print(param.__len__())
                # param.requires_grad=False
                #     print(param[1].requires_grad)
            # exit(0)

    def load_dataset(self):
        # 加载自定义数据集
        trainset = Dataer(r'/data1/yangyong/jqm/V1/data/train.txt')
        traindata = DataLoader(trainset, batch_size=self.batchsize, shuffle=True, drop_last=True)

        return traindata
    def init_write_log(self):
        flls=os.listdir('./log/train/')
        ls=[0]
        for i in flls:
            if 'exp' in i:
                ls.append(int(i.replace('exp','')))
        path=f'./log/train/exp{max(ls)+1}'
        os.mkdir(path)
        writer=SummaryWriter(path)
        return writer



if __name__ == '__main__':
    Yolo()



# model=torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50',pretrained=True)
# print(model)
# pre=model('../car.jpg')
# pre.print()
# pre.show()
