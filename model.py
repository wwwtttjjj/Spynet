from numpy import isin
import torch 
from torch import nn
import torch.nn.functional as F
import math
from utils import down_image
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class ConvBatchRelu(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding):
        super(ConvBatchRelu, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels=input,out_channels=output,kernel_size = kernel_size,stride=stride,padding=padding),
                                nn.BatchNorm2d(output),
                                nn.ReLU(inplace=True))
    def forward(self,x):
        return self.layer(x)

class Greate(nn.Module):
    def __init__(self,intLevel):
        super().__init__()
        self.layer1 = ConvBatchRelu(8, 32, 7, 1, 3)
        self.layer2 = ConvBatchRelu(32, 64, 7, 1, 3)
        self.layer3 = ConvBatchRelu(64, 32, 7, 1, 3)
        self.layer4 = ConvBatchRelu(32, 16, 7, 1, 3)
        self.layer5 = nn.Conv2d(16, 2, 7, 1, 3)
        self.netBasic = nn.Sequential(self.layer1, self.layer2,self.layer3, self.layer4, self.layer5)
    def forward(self, image_1, warp_image, flow_image):
        x = torch.cat([image_1, warp_image, flow_image], dim = 1)
        return self.netBasic(x)

backwarp_tenGrid = {}
def warp_feature(image_2, flow_image):
    if str(flow_image.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / flow_image.shape[3]), 1.0 - (1.0 / flow_image.shape[3]), flow_image.shape[3]).view(1, 1, 1, -1).repeat(1, 1, flow_image.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / flow_image.shape[2]), 1.0 - (1.0 / flow_image.shape[2]), flow_image.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, flow_image.shape[3])

        backwarp_tenGrid[str(flow_image.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    flow_image = torch.cat([ flow_image[:, 0:1, :, :] / ((image_2.shape[3] - 1.0) / 2.0), flow_image[:, 1:2, :, :] / ((image_2.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=image_2, grid=(backwarp_tenGrid[str(flow_image.shape)] + flow_image).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
# end



class SpyNet(nn.Module):
    def __init__(self, layers = 5):
        super(SpyNet, self).__init__()
        self.layers = layers
        self.netBasic = torch.nn.ModuleList([Greate(intLevel) for intLevel in range(self.layers)])

    # def deal_G_warp(self, image_1, image_2, flow_image, layer):
    #     if layer == 0:
    #         newflow_image = self.netBasic[layer](image_1, image_2, flow_image)
    #         return F.interpolate(newflow_image, scale_factor=2,mode = 'bilinear', align_corners=False)
    #     else:
    #         warp_image = warp_feature(image_2, flow_image)
    #         newflow_image = self.netBasic[layer](image_1, warp_image, flow_image)
    #         return F.interpolate(newflow_image, scale_factor=2,mode = 'bilinear', align_corners=False)

    def forward(self, image_1, image_2):
        image_1 = down_image(image_1, self.layers)
        image_2 = down_image(image_2, self.layers)

        flow_image = image_1[0].new_zeros([ image_1[0].shape[0], 2, int(math.floor(image_1[0].shape[2] / 2.0)), int(math.floor(image_1[0].shape[3] / 2.0)) ]).to(device)
        flow_image_arr = []
        for layer in range(self.layers):
            flow_image = torch.nn.functional.interpolate(input=flow_image, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow_image = self.netBasic[layer](image_1[layer], warp_feature(image_2[layer], flow_image), flow_image) + flow_image
            flow_image_arr.append(flow_image)
        return flow_image_arr
    def init_weight(self):
        for m in self.modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight,a=0,mode = 'fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU):
                torch.nn.init.constant_(m.weight, 1)

        

