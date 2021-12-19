#2021/11/21 21:15
import torch


def bbox_iou(box1, box2,x1y1x2y2=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # print(box2.shape)
    # box2 = box2.T
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
    inter = (torch.tensor(min(b1_x2, b2_x2)) - torch.tensor(max(b1_x1, b2_x1))).clamp(0) * \
            (torch.tensor(min(b1_y2, b2_y2)) - torch.tensor(max(b1_y1, b2_y1))).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union



    # if iou>offset:
    #     return True
    print(iou)
    return iou

def nms(data,offset=0.3):
    '''非极大值抑制'''
    res=[]
    # print(data)
    for ind,i in enumerate(data):
        tmp=i
        fl=False
        for ind0,j in enumerate(data):
            if j==i:continue
            else:
                if bbox_iou(i,j).item()>offset:
                    if tmp[-2]>j[-2]:
                        tmp=tmp
                    else:
                        tmp=j
                    fl=True

        if tmp not in res:
            res.append(tmp)

    return res



def return_xyxy(pre:torch.Tensor,confiden=0.6):
    '''计算所有大于阈值的预测框
    然后使用非极大值抑制选择出最终的预测框'''
    pre=torch.reshape(pre,(7,7,30))
    # pre_conf0_mask=pre[...,4]>confiden
    # pre_conf1_mask=pre[...,9]>confiden
    scale=1./7
    res=[]
    r,v=0,0
    for row in pre:
        v=0
        for i in row:
            _,pre=torch.max(i[10:], 0)

            # if i[4]*i[10+pre.item()]>confiden:
            if i[4]>confiden:
                x,y,w,h=i[:4]
                x,y,w,h=x.item(),y.item(),w.item(),h.item()
                x=(scale*v)+x*scale
                y=(scale*r)+y*scale
                x0,x1=x-w/2,x+w/2
                y0,y1=y-h/2,y+h/2
                res.append([x0,y0,x1,y1,i[4]*i[10+pre.item()],pre])
            # if i[9]*i[10+pre.item()]>confiden:
            if i[9]>confiden:
                x,y,w,h=i[5:9]
                x=(scale*v)+x*scale#,x+w/2
                y=(scale*r)+y*scale
                x0,x1=x-w/2,x+w/2
                y0,y1=y-h/2,y+h/2
                res.append([x0,y0,x1,y1,i[9]*i[10+pre.item()],pre])

            v+=1
        r+=1
    return nms(res)
    # return res



