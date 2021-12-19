#2021/11/13 22:19

from PIL import Image
from torchvision import transforms as tfs
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class lstm(nn.Module):
    def __init__(self, input_size=10, hidden_size=4, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x
import openpyxl
import pandas as pd
data=pd.read_excel(r'C:\Users\jqm\Documents\Tencent Files\1148679713\FileRecv\PM25.xls',)

# data=openpyxl.open(r'C:\Users\jqm\Documents\Tencent Files\1148679713\FileRecv\PM25.xlsx')
data=openpyxl.load_workbook(r'C:\Users\jqm\Documents\Tencent Files\1148679713\FileRecv\PM25.xlsx')
da=data.active
res=[]
for i in range(1,da.max_row+1):
    res.append(da.)

model = lstm(10, 4, 1, 2)
input=torch.rand((2,1,10))
aa=model(input)
print(aa.size())
print(aa[0])




# im = Image.open('./111.jpg')
# p1 = random.randint(0,1)
# p2 = random.randint(0,1)
# im_aug = transforms.Compose([
#                                   # transforms.RandomHorizontalFlip(p1),
#                                   # transforms.RandomVerticalFlip(p2),
#                                   # transforms.RandomRotation(10, resample=False, expand=False, center=None),
#                                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,hue=0.4,),
#                                   # transforms.RandomCrop(256),
#                                   transforms.Resize((512,512))
#                                        ])
# print(im_aug)
# seed = np.random.randint(2147483647)     # make a seed with numpy generator
#
# for i in range(5):
#     random.seed(seed)                    # apply this seed to img tranfsorms
#     f = plt.imshow(im_aug(im))
#     plt.show()





exit()









from model.yolo import Net
from torchvision.transforms import transforms
import cv2.cv2 as cv2
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from utils.data_loader import Dataer
import torch.optim as opt #import Optimizer
import torch
from utils.yolo_loss import Loss
import torchvision
import numpy as np
# import
from PIL import Image
import warnings
import math
warnings.filterwarnings('ignore')

print(math.ceil(0.001))
a=torch.tensor((1,6))
b=torch.tensor((3,4))
print(torch.min(a,b))

exit()


trans=transforms.Compose([transforms.ToTensor(),transforms.Resize((448,448))])
model=Net()
print(model)
# loss=nn.MSELoss()
loss=Loss()
opt=opt.SGD(model.parameters(),lr=0.001,momentum=0.8,weight_decay=0.0005)
img=cv2.imread('./data/tly.jpeg')
# img=cv2.resize(img,(448,448))
# cv2.imshow('img',img)
# img = np.asarray(img).transpose(-1, 0, 1)

# print(img.shape)
# cv2.waitKey(0)
img=trans(img)
# print(img.shape)
img=img.float().unsqueeze(0)
# print(img.shape)
opt.zero_grad()
out=model(img)
aa=loss(out,torch.tensor([0.5,0.3,0.2,0.6,1]))
aa.backward()
opt.step()

# print(out)