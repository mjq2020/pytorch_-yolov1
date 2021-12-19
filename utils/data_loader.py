#2021/11/14 21:47
import os
import cv2.cv2 as cv2
import numpy as np
import torch
from math import ceil
from torchvision.transforms import transforms
from numpy.random import choice
from PIL import Image
from torch.utils.data import Dataset



class Dataer(Dataset):
    def __init__(self,foldpath):
        # self.file=[os.path.join(i,) for i in os.listdir(foldpath)]
        # super(Dataer).__init__()
        self.trans=transforms.Compose([transforms.ToTensor(),transforms.Resize((448,448))])
        # self.trans=transforms.Compose([transforms.ToTensor(),transforms.Resize((448,448)),transforms.ColorJitter(0.4,0.4,0.4,0.4)])
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.data=open(foldpath,'r',encoding='utf8').read().split('\n')
        self.S,self.B,self.classes=7,7,20


    def __getitem__(self, item):
        file=self.data[item]
        # file=file.replace('D:\Code\Custom_YOLO','/data1/yangyong/jqm').replace('\\','/')

        label=self.read_label(file)
        # reslabels=np.zeros((self.S,self.S,self.classes+10))
        reslabels=torch.zeros(self.S,self.S,self.classes+10,dtype=torch.float32)
        # reslabels=torch.zeros(self.S,self.S,self.classes+10)
        # print(reslabels.shape)
        scale=1./self.S
        # print(label)
        # file = r'E:\BaiduDownloader\yolo\yolov1_pytorch-main\dog.jpg'
        # file = r'D:\Code\Custom_YOLO\V1\sudu.jpg'
        img = cv2.imread(file)

        imgshow=img.copy()
        # print(file)
        # print(imgshow.shape)
        yvalue,xvalue=imgshow.shape[:2]
        for l in label:
            x,y,w,h=l[1:5]
            xx0,xx1=int((x-(w/2))*xvalue),int((x+(w/2))*xvalue)
            yy0,yy1=int((y-(h/2))*yvalue),int((y+(h/2))*yvalue)
            cv2.rectangle(imgshow,(xx0,yy0),(xx1,yy1),color=(0,255,0),thickness=1)
            #这里有问题，xy对换
            row=ceil(y/scale)-1
            vol=ceil(x/scale)-1

            # x=((x-vol*scale)/scale)
            # y=((y-row*scale)/scale)

            reslabels[row,vol,4]=1
            reslabels[row,vol,9]=1
            reslabels[row,vol,10+int(l[0])]=1
            reslabels[row,vol,:4]=torch.tensor(l[1:])
            # reslabels[row,vol,:4]=torch.tensor([x,y,w,h])
            reslabels[row,vol,5:9]=torch.tensor(l[1:])
            # reslabels[row,vol,5:9]=torch.tensor([x,y,w,h])
        # cv2.imwrite('D:\Code\Custom_YOLO\V1\log\img\{0}'.format(file.split('\\')[-1]),imgshow)
        cv2.imshow('img',imgshow)
        # cv2.waitKey(0)
        # im = cv2.resize(img, (int(w0 * r), int(h0 * r)),
        #                 interpolation=cv2.INTER_AREA if r < 1 and not False else cv2.INTER_LINEAR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255

        im=self.trans(img).to(torch.float32)
        # reslabels=transforms.ToTensor()(reslabels)
        # print(reslabels.shape)
        # print(im)
        # print(reslabels[...,:10])
        # print(reslabels.shape)
        # exit(0)
        # torch.set_printoptions(precision=8)
        # print(reslabels[3][0])
        return im,reslabels




    def __len__(self):
        return len(self.data)


    def read_label(self,file:str):
        '''返回图片中的类别以及每个类别的 x y w h'''
        file=file.replace('jpg','txt').replace('images','labels')
        data=open(file,'r',encoding='utf8').read().split('\n')
        data.remove('')
        data = [self.reint(i.split(' ')) for i in data]
        # print(data)
        return np.array(data)

        return transforms.ToTensor()(np.array(data,dtype=np.float64))

    def reint(self,ls):
        # print(ls)
        return list(map(float,ls))


if __name__ == '__main__':
    data=Dataer(r'D:\Code\Custom_YOLO\V1\data\train.txt')
    data.__getitem__(np.random.randint(1,500))