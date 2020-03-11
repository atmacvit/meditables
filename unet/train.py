import torch
import torch.nn as nn
import cv2
import torch.optim as optim
import time
import copy
from model import UNet
import json
import torch.nn.functional as F
from loss import loss_fn
from data import UnetDataset
from utils import weights_init
import numpy as np
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms,utils
from torchsummary import summary
from torch.utils.data import DataLoader



#Training Arguments
parser = argparse.ArgumentParser(description='For Getting Training Arguments')
parser.add_argument('--image_dir', type=str,
                    help='Path to Training Images')
parser.add_argument('--label_dir',type=str,help='Path to Training Labels')
parser.add_argument('--batch_size',type=int,default=1,help='Size of training batch')
parser.add_argument('--num_class',type=int,default=1,help='Num of classes in Training Data')
parser.add_argument('--num_epoch',type=int,default=60,help='Number of epoch to Train the Model for')
parser.add_argument('--lr',type=float,default = 0.001,help="Learning Rate for Training")
parser.add_argument('--multi_gpu',type=bool,default = False,help="Flag for Multi Gpu Training")
parser = parser.parse_args()

args = vars(parser)

print('------------ Training Args -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('----------------------------------------')


assert os.path.exists(args["image_dir"])
assert os.path.exists(args["label_dir"])



train_transforms = transforms.Compose([
    transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on : {}".format(device))
#

train_data = UnetDataset(args["image_dir"],args["label_dir"],transform = train_transforms)
trainloader = DataLoader(train_data,batch_size =4,shuffle=True)

print(train_data.__len__())
print(train_data.__getitem__(4)[0].shape)


if not os.path.exists("./outputs"):
    os.mkdir("output")


net = UNet(args["num_class"]).to(device)
summary(net,(1,512,512))
optimizer = optim.Adam(net.parameters(), lr=2e-4)


if args["multi_gpu"] == True:
    net = nn.DataParallel(net).to(device)

for epoch in range(args["num_epoch"]):
    epoch_loss = 0

    epoch_samples = 0
    for batch_idx,batch in enumerate(trainloader):
          net.train()
          img = Variable(batch[0]).to(device)
          gt_mask = Variable(batch[1]).to(device)
          epoch_samples += img.size(0)
          pred_mask = net(img)
          loss_model = loss_fn(gt_mask,pred_mask,epoch)
          epoch_loss += loss_model
          print("Epoch : {} || Batch Id: {} || Loss: {}".format(epoch,batch_idx,loss_model.mean().item()))
          optimizer.zero_grad()
          loss_model.backward()
          optimizer.step()
    print("Epoch Loss for Epoch : {} is {} ".format(epoch,epoch_loss))
    torch.save(net.state_dict(),"./output/checkkpoints_{}.pth".format(epoch))
