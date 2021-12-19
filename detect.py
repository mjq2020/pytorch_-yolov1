#2021/11/21 20:32

import torch
import torchvision
import cv2.cv2 as cv2
import warnings
from PIL import Image
import torch.nn as nn
from model.yolo import Net
from utils.data_loader import Dataer
from torch.utils.data import DataLoader
from utils.computer_xyxy import return_xyxy
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


def load_model(device):
    model=Net()

    model.load_state_dict(torch.load(r'D:\Code\Custom_YOLO\V1\weights\best11.pt',map_location=device))
    return model


def load_resnet(device):
    model=torchvision.models.resnet152(pretrained=False)
    model.layer4.add_module('conver1', nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
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
                                                     nn.LeakyReLU(negative_slope=0.1)))

    model.fc = nn.Sequential(
        nn.Linear(in_features=1024, out_features=2048, bias=True),
        # nn.BatchNorm1d(1),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=2048, out_features=1470, bias=True),
        nn.Sigmoid(),


    )
    # file=torch.load(r'D:\Code\Custom_YOLO\V1\weights\best11.pt',map_location=device)
    model.load_state_dict(torch.load(r'D:\Code\Custom_YOLO\V1\weights\best11.pt',map_location=device))
    return model

classer=['dog', 'person', 'train', 'sofa', 'chair', 'car', 'pottedplant', 'diningtable', 'horse', 'cat', 'cow', 'bus', 'bicycle', 'aeroplane', 'motorbike', 'tvmonitor', 'bird', 'bottle', 'boat', 'sheep']


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=load_resnet(device)
model.to(device)
model.eval()
datatest=Dataer('./data/test.txt')
testset=DataLoader(datatest,shuffle=True)
# testset=DataLoader(datatest)
model.eval()

for data in testset:
    img,labels = data

    img,labels =img.to(device),labels.to(device)
    # print(img.shape)
    # print(labels.shape)
    pre=model(img)

    xyxy=return_xyxy(pre,0.2)

    img=img.permute(0,2,3,1)
    # print(xyxy)

    img=img.squeeze(0).numpy()
    # print(img.shape)
    # img=transforms.ToPILImage()(img.squeeze(0))
    # Image._show(img)
    # Image.open(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    lw = max(round(sum(img.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)
    # print(type(img))
    xvalue,yvalue=448,448


    for i in xyxy:
        x0, y0, x1, y1 = int(i[0] * xvalue), int(i[1] * yvalue), int(i[2] * xvalue), int(i[3] * yvalue)

        w, h = cv2.getTextSize(classer[i[-1].item()], 0, fontScale=lw / 3, thickness=tf)[0]
        outside = y0 - h - 3 >= 0  # label fits outside box

        # print(x0,y0,x1,y1)
        cv2.rectangle(img,(x0,y0),(x1,y1),color=(0,255,0),thickness=1)
        cv2.putText(img,classer[i[-1].item()],(x0,y0-2 if outside else y0+h+2),0,lw / 3,color=(0,255,0),thickness=tf,lineType=cv2.LINE_AA)
        # print(classer[i[-1]])
    # (p1[0], p1[1] - 2 if outside else p1[1] + h + 2)
    cv2.imshow('pre_img',img)
    cv2.waitKey(0)



