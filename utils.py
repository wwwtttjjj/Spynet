import torch 
from torch import nn
import torchvision
import torch.nn.functional as F

def down_image(image, num):
    down_imageslist = []
    for i in range(num):
        down_imageslist.append(image)
        image = F.interpolate(image, scale_factor=1/2,mode = 'bilinear', align_corners=False)
    return down_imageslist[::-1]


class Transform:

    def __init__(self):
        self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(), 
                torchvision.transforms.Resize((448, 1024)),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
        ])
        self.transformRs = torchvision.transforms.Compose([
                # torchvision.transforms.ToTensor(), 
                torchvision.transforms.Resize((448, 1024)),  
        ])
    def __call__(self, img_1,img_2,flow,valid_flow_mask):
        img_1 = self.transform(img_1)
        img_2 = self.transform(img_2)
        flow = torch.tensor(flow)
        flow = self.transformRs(flow)
        return (img_1,img_2,flow,valid_flow_mask)