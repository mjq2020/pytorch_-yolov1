#2021/11/14 22:36
import os
import numpy as np
import random

# np.random.choice()

path='../data/images'
filels=os.listdir(path)

def splitdata(filels,trainlr=0.98):
    train=np.random.choice(filels,size=int(len(filels)*trainlr),replace=False)
    valtest=list(set(filels)-set(train))
    # k=len(valtest)//2
    # val=np.random.choice(valtest,size=k,replace=False)
    # test=list(set(valtest)-set(val))

    # print(len(train))
    # print(len(val))
    # print(len(test))
    return train,valtest

def write_data(ls,name):

    with open(f'../data//{name}.txt','w',encoding='utf8')as f:
        for i in ls:
            if i==ls[-1]:f.write(f'{i}')
            else:
                f.write(f'{i}\n')

abspath = os.getcwd()

abspath=abspath.replace(abspath.split('\\')[-1],'data/images').strip('\\')

fileall=[os.path.join(abspath,i) for i in filels]

train,val=splitdata(fileall)
write_data(train,'train')
# write_data(val,'val')
write_data(val,'test')
