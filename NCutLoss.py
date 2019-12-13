# From https://github.com/Andrew-booler/W-Net/

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function
import time
import pdb
import subprocess
import numpy as np

from config import Config

config = Config()

# The weight matrix w is a measure of the weight between each pixel and
# every other pixel. so w[u][v] is a measure of
#   (a) Distance between the brightness of the two pixels.
#   (b) Distance in positon between the two pixels

# The NCut loss metric is then:
#   (a) association:    sum(weight_connection) * P(both pixels in the connection are in the class)
#   (b) disassociation: sum(weight_connection) * P(first pixel is in the class)
#   N Cut loss = disassociation / association

class NCutsLoss(nn.Module):
    def __init__(self):
        super(NCutsLoss,self).__init__()

    def forward(self, seg, padded_seg, weight,sum_weight):
        cropped_seg = []
        for m in torch.arange((config.radius-1)*2+1,dtype=torch.long):
            column = []
            for n in torch.arange((config.radius-1)*2+1,dtype=torch.long):
                column.append(padded_seg[:,:,m:m+seg.size()[2],n:n+seg.size()[3]].clone())
            cropped_seg.append(torch.stack(column,4))
        cropped_seg = torch.stack(cropped_seg,4)
        #for m in torch.arange(50,70,dtype=torch.long):

        #    print(m)
        #    for n in torch.arange(50,70,dtype= torch.long):
        #        print(weight[5,0,m,n])
        multi1 = cropped_seg.mul(weight)
        multi2 = multi1.sum(-1).sum(-1).mul(seg)
        multi3 = sum_weight.mul(seg)
        #print("=============================================================================")
        #for a in [0,1]:
        #    print(multi2[5,0,a*10+50:a*10+60,50:60])
        #    print(multi2[5,0,a*10+50:a*10+60,60:70])
        assocA = multi2.view(multi2.shape[0],multi2.shape[1],-1).sum(-1)
        assocV = multi3.view(multi3.shape[0],multi3.shape[1],-1).sum(-1)
        assoc = assocA.div(assocV).sum(-1)

        return torch.add(-assoc,config.k)
