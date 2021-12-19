#2021/11/14 10:41
import torch
import torch.nn.functional as F
import torch.nn as nn
from math import ceil
import numpy as np
class Loss():
    def __init__(self):
        super(Loss,self).__init__()
        self.scale=torch.tensor(1.0/7)

    def computer_loss(self,pre,target):
        '''计算均方误差'''
        return F.mse_loss(pre,target,reduction='sum')

    def __call__(self,pre:torch.tensor,target:torch.tensor):
        torch.autograd.set_detect_anomaly(True)
        # a=torch.FloatTensor([0.8])
        # b=torch.FloatTensor([0.5])
        # print(F.mse_loss(torch.sqrt_(a),torch.sqrt_(b)))

        # return F.mse_loss(pre,target)



        batch=pre.shape[0]
        pre = torch.reshape(pre, (batch, 7, 7, 30))
        #先找出掩码
        target_grid_mask=(target[...,4]>0).unsqueeze(-1).expand_as(target)
        target_nogrid_mask=(target[...,4]==0).unsqueeze(-1).expand_as(target)

        target_grid_mask=target_grid_mask.bool()
        target_nogrid_mask=target_nogrid_mask.bool()


        target_grid_value=target[target_grid_mask].view(-1,30)
        target_nogrid_value=target[target_nogrid_mask].view(-1,30)
        # print(target_grid_value[0])


        pre_grid_value=pre[target_grid_mask].view(-1,30)
        pre_nogrid_value=pre[target_nogrid_mask].view(-1,30)
        # print(pre_grid_value[0])

        #
        # loss_class=self.computer_loss(pre_grid_value[...,10:],target_grid_value[...,10:])
        #loss_class=F.mse_loss(pre_grid_value[...,10:],target_grid_value[...,10:],reduction='sum')
        # return loss_class

        target_bbox_xywhc=target_grid_value[:,:10].contiguous().view(-1,5)
        target_nobbox_xywhc=target_nogrid_value[:,:10].contiguous().view(-1,5)

        pre_bbox_xywhc=pre_grid_value[:,:10].contiguous().view(-1,5)
        pre_nobbox_xywhc=pre_nogrid_value[:,:10].contiguous().view(-1,5)

        # # loss_xy=torch.tensor([0.],dtype=torch.float32)
        # loss_xy=None
        # loss_wh=torch.tensor([0.],dtype=torch.float32)
        # loss_conf=torch.tensor([0.],dtype=torch.float32)
        # loss_npbbpx_conf=torch.tensor([0.],dtype=torch.float32)

        target_mse_xywhc=torch.BoolTensor(target_bbox_xywhc.size()).fill_(0)
        target_nomse_xywhc=torch.BoolTensor(pre_bbox_xywhc.size()).fill_(0)
        target_mse_iou=torch.BoolTensor(pre_bbox_xywhc.size())


        for i in range(0,target_bbox_xywhc.size()[0],2):
            target_bb0=target_bbox_xywhc[i]
            target_bb1=target_bbox_xywhc[i+1]
            pre_bb0=pre_bbox_xywhc[i]
            pre_bb1=pre_bbox_xywhc[i+1]

            iou0=self.bbox_iou(target_bb0[:4],pre_bb0[:4])
            iou1=self.bbox_iou(target_bb1[:4],pre_bb1[:4])
            # print(target_bbox_xywhc[i,:2])
            if iou0>iou1:
                target_mse_xywhc[i]=1
                target_nomse_xywhc[i+1]=1
                target_bbox_xywhc[i+1,4]=0
                xy=target_bbox_xywhc[i,:2]

                target_bbox_xywhc[i,:2]=(xy-((xy/self.scale).ceil()-1)*self.scale)/self.scale

            else:
                target_mse_xywhc[i+1]=1
                target_nomse_xywhc[i]=1
                target_bbox_xywhc[i, 4] = 0
                xy=target_bbox_xywhc[i+1, :2]
                target_bbox_xywhc[i+1, :2]=(xy-((xy/self.scale).ceil()-1)*self.scale)/self.scale
            # print(target_bbox_xywhc[i,:2])
            # print(target_bbox_xywhc[i])

            # target_bbox_xywhc[i+1,:2] = target_bbox_xywhc[i, :2] = (target_bbox_xywhc[i, :2]-((target_bbox_xywhc[i, :2]/self.scale).ceil()-1)*self.scale)/self.scale
            # print(target_bbox_xywhc[i+1])
        target_nobbox_mse=target_bbox_xywhc[target_nomse_xywhc].view(-1,5)
        pre_nobbox_mse=pre_bbox_xywhc[target_nomse_xywhc].view(-1,5)

        target_bbox_mse=target_bbox_xywhc[target_mse_xywhc].view(-1,5)
        pre_bbox_mse=pre_bbox_xywhc[target_mse_xywhc].view(-1,5)

        loss_xy=F.mse_loss(pre_bbox_mse[:,:2],target_bbox_mse[:,:2],reduction='sum')
        loss_wh=F.mse_loss(torch.sqrt(pre_bbox_mse[:,2:4]),torch.sqrt(target_bbox_mse[:,2:4]),reduction='sum')
        loss_conf=F.mse_loss(pre_bbox_mse[:,4],target_bbox_mse[:,4],reduction='sum')
        loss_noconf=F.mse_loss(pre_nobbox_mse[:,4],target_nobbox_mse[:,4],reduction='sum')
        loss_noconf=loss_noconf+F.mse_loss(pre_nobbox_xywhc[:,4],target_nobbox_xywhc[:,4],reduction='sum')
        loss_class=F.mse_loss(pre_grid_value[...,10:],target_grid_value[...,10:],reduction='sum')

        # print(loss_xy)
        # print(loss_wh)
        # print(loss_class)
        # print(loss_conf)
        # print(loss_npbbpx_conf)
        # return loss_xy
        return loss_class/batch
        loss=5.*(loss_xy+loss_wh)+0.5*loss_noconf+loss_class+loss_conf
        return loss/batch


    def backward(self,ind):
        return ind

    def forward1(self,x:torch.Tensor,labels:torch.Tensor):
        # print('++++++++++++开始计算损失++++++++++++')
        # print('预测大小：',x.shape)
        # print('标签大小：',labels.shape)
        loss=0
        b = 1/7
        #获取 batch size
        batch_size=labels.size()[0]
        #含有目标的grid 的掩码
        coord_mask=labels[...,4]==1
        #不含目标的grid 的掩码
        noobj_mask=labels[...,4]==0

        #将掩码的维度拓展到和标签维度一致
        coord_mask=coord_mask.unsqueeze(-1).expand_as(labels)
        noobj_mask=noobj_mask.unsqueeze(-1).expand_as(labels)


        #将张量内的值转换为bool
        coord_mask=coord_mask.bool()
        noobj_mask=noobj_mask.bool()

        # print(coord_mask.shape)
        # print(noobj_mask.shape)
        # coord_obj=coord_obj.bool()
        # noodb_obj=noodb_obj.bool()
        #提取预测值中含有目标中心点的grid
        coored_pred = x[coord_mask].view(-1,30)
        print(coored_pred.shape)
        #提取预测bbox中的xywh值及confident，并将每行为一个bbox
        bbox_pred = coored_pred[:, :10].contiguous().view(-1, 5)
        # print(bbox_pred.shape)
        #提取预测结果中的分类信息
        bbox_class = coored_pred[:, 10:].contiguous()
        # print(bbox_class.shape)

        #提取标签中含有标签的grid
        coored_label=labels[coord_mask].view(-1,30)

        #提取标签中xywh及confident
        bbox_label=coored_label[:,:10].contiguous().view(-1,5)
        #提取标签中的分类信息
        class_label=coored_label[:,10:]

        #对没有目标的值进行处理
        #提取预测值中没有目标的gird   一共是batchsize*49个grid
        noobj_pred=x[noobj_mask].view(-1,30)
        print(noobj_pred.shape)
        #提取标签中没有目标的张量，grid
        noobj_label=labels[noobj_mask].view(-1,30)
        # print(noobj_pred.shape)

        noobj_conf_mask=torch.BoolTensor(noobj_pred.size()).fill_(0)
        # print(noobj_conf_mask[0])
        # print(noobj_conf_mask.shape)

        for b in range(2):
            noobj_conf_mask[:,4+b*5] = 1
        # print(noobj_conf_mask[0])
        # print(noobj_conf_mask.shape)

        noobj_pred_conf=noobj_pred[noobj_conf_mask]
        noobj_label_conf=noobj_label[noobj_conf_mask]
        # print(noobj_pred_conf)
        # print(noobj_pred_conf.shape)
        # print(noobj_label_conf)
        #求没有目标的置信度loss
        loss_noobj=F.mse_loss(noobj_pred_conf,noobj_label_conf,reduction='sum')


        coord_xywh_mask=torch.BoolTensor(bbox_label.size()).fill_(0)
        coord_not_xywh_mask=torch.BoolTensor(bbox_label.size()).fill_(1)
        # print(coord_xywh_mask)
        # print(coord_not_xywh_mask)
        # print(coord_xywh_mask.shape)

        bbox_label_iou=torch.zeros(bbox_label.size())
        # print(bbox_label_iou.shape)
        for i in range(0,bbox_label.size()[0],2):
            iou1=self.bbox_iou(bbox_pred[i,:4],bbox_label[i,:4])
            iou2=self.bbox_iou(bbox_pred[i+1,:4],bbox_label[i+1,:4])
            coord_xywh_mask[i],coord_not_xywh_mask=(1,0) if iou1>iou2 else (0,1)
            if iou1>iou2:
                bbox_label_iou[i,4]=iou1
            else:bbox_label_iou[i+1]=iou2

        bbox_pred_response=bbox_pred[coord_xywh_mask].view(-1,5)
        bbox_target_response=bbox_label[coord_xywh_mask].view(-1,5)
        target_iou=bbox_label_iou[coord_xywh_mask].view(-1,5)
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')




        # F.mse_loss(bbox_pred[:,:2],bbox_label[:,:2])
        # F.mse_loss(torch.sqrt(bbox_pred[:,2:4]),bbox_label[:,2:4])

        loss_class=F.mse_loss(bbox_class,class_label,reduction='sum')

        loss=5*(loss_xy+loss_wh)+0.5*loss_noobj+loss_class+loss_obj
        loss.requires_grad=True
        return loss+pre-pre


        return torch.autograd.variable(loss,requires_grad=True)/batch_size



        return torch.tensor([1,2,3,4],requires_grad=True,dtype=torch.float32).view(4,1)
        # return torch.autograd.Variable(torch.tensor([1],requires_grad=True,dtype=torch.float32),requires_grad=True)

        coored_pred=x.unsqueeze(0)[coord_obj].view(-1,30)
        print(coored_pred)
        print(coored_pred.shape)
        bbox_pred=coored_pred[:,:10].contiguous().view(-1,5)
        print(bbox_pred)
        print(bbox_pred.shape)
        bbox_class=coored_pred[:,10:].contiguous()
        print(bbox_class)
        print(bbox_class.shape)
        xx=labels[0].item()
        y=labels[1].item()
        w=labels[2].item()
        h=labels[3].item()

        l=labels[4]
        self.volum=ceil(xx/b)
        self.rows=ceil(y/b)

        gridx=xx/b-xx//b
        gridy=y/b-y//b
        losswh=0
        lossxy=0
        #遍历7*7的grid
        for i in range(7):
            for j in range(7):
                iou0 = self.iou(x[i, j, :5], labels)

                iou1 = self.iou(x[i, j, 5:10], labels)
                confdence0 = x[i, j, 4]
                confdence1 = x[i, j, 9]

                if j==self.volum and i == self.rows:

                    x0 = (x[i,j,0] if x[i,j,0] >= 0 else 0) + self.volum * b
                    y0 = (x[i,j,1] if x[i,j,1] >= 0 else 0) + self.rows * b

                    x1 = (x[i,j,5] if x[i,j,5] >= 0 else 0) + self.volum * b
                    y1 = (x[i,j,6] if x[i,j,6] >= 0 else 0) + self.rows * b


                    w0=x[i,j,2]
                    w1=x[i,j,7]

                    h0=x[i,j,2]
                    h1=x[i,j,7]
                    # 检测物体的置信度误差
                    loss+=((confdence0-iou0) if iou0>iou1 else (confdence1-iou1))**2
                    # 检测物体的bbox的宽高误差 损失
                    losswh+=(((w0-w)**2+(h0-h)**2) if iou0>iou1 else ((w1-w)**2+(h1-h)**2))
                    #负责检测物体bbox的中心定位误差
                    lossxy+=(((x0-xx)**2+(y0-y)**2) if iou0>iou1 else ((x1-xx)**2+(y1-y)**2))

                    #负责检测grid的分类误差 损失
                    for k in range(20):
                        if k==l:
                            loss += torch.pow(x[i, j, 10:][k]-1, 2).item()
                        else:
                            loss+=torch.pow(0-x[i,j,10:][k]-0,2).item()
                else:
                    #不负责检测物体的bbox的置信度误差
                    loss+= ((confdence0 - 0)**2)
                    loss+= ((confdence1 - 0) ** 2)


        loss=lossxy+losswh+loss
        print('LOSS:',loss)
        return loss
        # return torch.tensor([loss],dtype=torch.float32)

    def iou(self,x1:torch.Tensor,x2:torch.Tensor):

        # b=1/7
        #
        #
        # x1[0]=(x1[0] if x1[0]>=0 else 0) + self.volum*b
        # x1[1]=(x1[1] if x1[1]>=0 else 0) + self.rows*b


        return self.bbox_iou(x1,x2,False)


        # xx2=x2[0]*b
        # xy2=x2[1]*b
        #
        # xx1=x1[0] if x1[1]>=0 else 0  *b
        # xy1=x1[1] if x2[1]>=0 else 0 *b
        #
        # w1=x1[2] if x1[2]>=0 else 0 *448
        # h1=x1[3] if x1[3]>=0 else 0 *448
        #
        # w2=x2[2] if x1[2]>=0 else 0 *448
        # h2=x2[3] if x1[3]>=0 else 0 *448
        #
        # xcha=abs(xx1-xx2)
        # ycha=abs(xy1-xy2)
        #
        # #重合面积
        # wu=(w1+w2-xcha)/2
        # hu=(h1+h2-ycha)/2



    def bbox_iou(self,box1, box2, x1y1x2y2=False,eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box2 = box2.T

        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        # print(iou)
        return iou
