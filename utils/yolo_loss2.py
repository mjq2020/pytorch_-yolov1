#2021/11/25 14:18
#2021/11/14 10:41
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from math import ceil
import numpy as np
class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        self.scale=torch.tensor(1.0/7)

    def computer_loss(self,pre,target):
        '''计算均方误差'''
        return F.mse_loss(pre,target,reduction='sum')

    def forward(self,pre:torch.tensor,target:torch.tensor):
        device=pre.device
        eps=torch.autograd.Variable(torch.tensor(1e-7))
        torch.autograd.set_detect_anomaly(True)

        batch=pre.shape[0]
        pre = torch.reshape(pre, (batch, 7, 7, 30))

        #先找出掩码
        target_grid_mask=(target[...,4]>0).unsqueeze(-1).expand_as(target)
        target_nogrid_mask=(target[...,4]==0).unsqueeze(-1).expand_as(target)

        target_grid_mask=target_grid_mask.bool()
        target_nogrid_mask=target_nogrid_mask.bool()

        #和掩码与操作，获取有目标的grid和没目标的grid
        target_grid_value=target[target_grid_mask].view(-1,30)
        target_nogrid_value=target[target_nogrid_mask].view(-1,30)

        pre_grid_value=pre[target_grid_mask].view(-1,30)
        pre_nogrid_value=pre[target_nogrid_mask].view(-1,30)

        #将获取到的grid的值进行重新排列，（获取前10的数据）xywhc
        target_bbox_xywhc=target_grid_value[:,:10].contiguous().view(-1,5)
        target_nobbox_xywhc=target_nogrid_value[:,:10].contiguous().view(-1,5)



        pre_bbox_xywhc=pre_grid_value[:,:10].contiguous().view(-1,5)
        pre_nobbox_xywhc=pre_nogrid_value[:,:10].contiguous().view(-1,5)

        # print('处理前10项：')
        target_mse_xywhc=torch.BoolTensor(target_bbox_xywhc.size()).fill_(0).to(device)
        target_nomse_xywhc=torch.BoolTensor(pre_bbox_xywhc.size()).fill_(1).to(device)
        # target_mse_iou=torch.BoolTensor(pre_bbox_xywhc.size())
        # print(target_nomse_xywhc)
        # print(target_mse_xywhc.shape)
        # print(target_nomse_xywhc.shape)


        for i in range(0,target_bbox_xywhc.size()[0],2):
            # target_bb0=target_bbox_xywhc[i]
            #
            # target_bb1=target_bbox_xywhc[i+1]
            # pre_bb0=pre_bbox_xywhc[i]
            # pre_bb1=pre_bbox_xywhc[i+1]

            # print((target_bb0[:,:4],pre_bb0[:,:4]))
            # iou0=self.bbox_iou(pre_bb0[:4],target_bb0[:4])
            # iou1=self.bbox_iou(pre_bb1[:4],target_bb1[:4])
            # print('------------')
            # print(target_bb0)
            # print(target_bb1)
            # print('Only:')
            # print(iou0)
            # print(iou1)
            target_bb0 = target_bbox_xywhc[i:i+2]
            pre_bb0 = pre_bbox_xywhc[i:i+2]


            iou0=self.bboxs_iou(pre_bb0[:,:4],target_bb0[:,:4])
            # iou1=self.bbox_iou(pre_bb1[:4],target_bb1[:4])

            max_iou,max_index=iou0.max(0)
            min_iou,min_index=iou0.min(0)
            # print('Double:',iou0)
            print('------------')
            print('max:',max_index)
            print('min:',min_index)
            ind=max_index.data.to(device)
            ind0=min_index.data.to(device)
            target_mse_xywhc[i+ind]=1
            target_nomse_xywhc[i+ind]=0
            target_bbox_xywhc[i+ind0,4]=torch.autograd.Variable(torch.tensor(0))
            xy=target_bbox_xywhc[i+ind0, :2]
            target_bbox_xywhc[i+ind,:2] = (xy-((xy/self.scale).ceil()-1)*self.scale)/self.scale


        target_nobbox_mse=target_bbox_xywhc[target_nomse_xywhc].view(-1,5)
        pre_nobbox_mse=pre_bbox_xywhc[target_nomse_xywhc].view(-1,5)
        # print("TARGET")


        target_bbox_mse=target_bbox_xywhc[target_mse_xywhc].view(-1,5)
        pre_bbox_mse=pre_bbox_xywhc[target_mse_xywhc].view(-1,5)

        loss_xy=F.mse_loss(pre_bbox_mse[:,:2],target_bbox_mse[:,:2],reduction='sum')
        print('==========')
        print(pre_bbox_mse[:,4])
        print(target_bbox_mse[:,4])
        print(pre_bbox_mse[:,4])
        print(target_nobbox_mse[...,4])
        print(target_nobbox_xywhc[...,4])
        print('==========')
        loss_wh=F.mse_loss(torch.sqrt(pre_bbox_mse[:,2:4]+eps),torch.sqrt(target_bbox_mse[:,2:4]+eps),reduction='sum')
        loss_conf=F.mse_loss(pre_bbox_mse[:,4],target_bbox_mse[:,4],reduction='sum')



        loss_noconf=F.mse_loss(pre_nobbox_mse[...,4],target_nobbox_mse[...,4],reduction='sum')
        loss_noconf=loss_noconf+F.mse_loss(pre_nobbox_xywhc[...,4],target_nobbox_xywhc[...,4],reduction='sum')

        loss_class = F.mse_loss(pre_grid_value[..., 10:], target_grid_value[..., 10:], reduction='sum')



        loss=5.*(loss_xy+loss_wh)+loss_class+loss_conf+0.5*loss_noconf

        data = {}
        data['Sum'] = loss/float(batch)
        data['Class']=loss_class
        data['XY']=loss_xy
        data['WH']=loss_wh
        data['Conf']=loss_conf
        data['NoConf']=loss_noconf

        return loss/float(batch),data


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
        # print(coored_pred.shape)
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
        # print(noobj_pred.shape)
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
        # print(bbox_pred_response[:, 4])

        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')




        # F.mse_loss(bbox_pred[:,:2],bbox_label[:,:2])
        # F.mse_loss(torch.sqrt(bbox_pred[:,2:4]),bbox_label[:,2:4])

        loss_class=F.mse_loss(bbox_class,class_label,reduction='sum')

        loss=5*(loss_xy+loss_wh)+0.5*loss_noobj+loss_class+loss_obj
        loss.requires_grad=True
        return loss



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
    def bboxs_iou(self,box1, box2, x1y1x2y2=False,eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        # print(box2.shape)
        # box2 = box2.T
        # print(box2.shape)
        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[:,0] - box1[:,2] / 2, box1[:,0] + box1[:,2] / 2
            b1_y1, b1_y2 = box1[:,1] - box1[:,3] / 2, box1[:,1] + box1[:,3] / 2
            b2_x1, b2_x2 = box2[:,0] - box2[:,2] / 2, box2[:,0] + box2[:,2] / 2
            b2_y1, b2_y2 = box2[:,1] - box2[:,3] / 2, box2[:,1] + box2[:,3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)+eps

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        # print('IOU:',union)
        # print('INTER:',inter)
        iou = inter / union
        print(iou)
        return iou


    def bbox_iou(self,box1, box2, x1y1x2y2=False,eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        # print(box2.shape)
        box2 = box2.T
        # print(box2.shape)
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
    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou