import torch
import torchvision
from model import SpyNet
import wandb
from utils import Transform, down_image
import math
layer = 5

mytransform = Transform()


train_image = torchvision.datasets.Sintel(root = 'data/',split="train", pass_name='clean', transforms=mytransform)
test_image = torchvision.datasets.Sintel(root = 'data/', split='test', pass_name='clean', transforms=mytransform)

train_iter = torch.utils.data.DataLoader(train_image, batch_size=1,shuffle=True,drop_last=True)
test_iter = torch.utils.data.DataLoader(test_image, batch_size=8,shuffle=False,drop_last=True)


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
epochs = 50

model = SpyNet(layer)

optimizer = torch.optim.Adam([{'params':model.netBasic[-1].parameters(), 'lr':1e-3},
                            {'params':model.netBasic[:-1].parameters(), 'lr':1e-4}
                            ],lr = 1e-4,betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
[]
wandb.init(project='spynet',entity='wtj')
wandb.config = {
    'learning_rate':0.0001,
    'epochs' :50,
    'batch_size' :8
}

model.to(device)
model.train()


for epoch in range(epochs):
    loss_arr = []
    for x_cpu in train_iter:
        optimizer.zero_grad()
        image_1 = x_cpu[0].to(device)
        image_2 = x_cpu[1].to(device)
        flow_image = down_image(x_cpu[2], layer)
        flowpre_image = model(image_1, image_2)
        for l in range(layer):
            pre_image = flowpre_image[l].to(device)
            ground_image = flow_image[l].to(device)
            print(pre_image.shape, ground_image.shape)
            loss_arr.append(torch.linalg.norm((ground_image-pre_image), ord=2, dim = 1).mean())
        print(loss_arr)
        loss = torch.tensor(loss_arr).sum()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        # print(flow_image.shape, flowpre_image.shape)
torch.save(model, 'pth/spynet.pth')

        