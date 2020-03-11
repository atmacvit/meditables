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





net = UNet(args["num_class"]).to(device)
summary(net,(1,512,512))
optimizer = optim.Adam(net.parameters(), lr=2e-4)


if args["multi_gpu"] == True:
    net = nn.DataParallel(net).to(device)

for epoch in range(args["num_epoch"]):
#     epoch_loss = 0
#     train_loss = []
#     val_loss = []
#     epoch_train_loss = []
#     epoch_val_loss = []
    epoch_samples = 0
# # #    print("Epoch Loss for Epoch : {} is {} ".format(epoch,epoch_loss))
    for batch_idx,batch in enumerate(trainloader):
          print("Batch id: {}".format(batch_idx))
          net.train()
          img = Variable(batch[0]).to(device)
          gt_mask = Variable(batch[1]).to(device)
          epoch_samples += img.size(0)
          pred_mask = net(img)
          # loss_model = criterion(gt_mask,pred_mask,epoch)
          loss_model = loss_fn(gt_mask,pred_mask,epoch)
          print(loss_model.mean().item())
# #           #loss_model = criterion(pred_mask, mask)
#           loss_model = loss_model.mean()
#           epoch_loss += loss_model.item()
          optimizer.zero_grad()
          loss_model.backward()
          optimizer.step()
# # #          print("EPOCH:{0} || BATCH NO:{1} || LOSS:{2}".format(epoch,batch_idx,loss_model.item()))
#           if batch_idx%3000 == 0:
#               torch.save(model.module.state_dict(),"../outputs12/checkpoints/ckpt_{}_{}.pth".format(batch_idx,epoch))
#               metrics["batch_idx"] = batch_idx
#               metrics["epoch"] = epoch
#               metrics["epoch_samples"] = epoch_samples
#               a = open("../outputs12/train_logs/train_metrics_{}_{}.json".format(batch_idx,epoch), 'w+')
#               with open("../outputs12/train_logs/train_metrics_{}_{}.json".format(batch_idx,epoch), 'w') as wrf:
#                   json.dump(metrics, wrf)
#               for idx in range(img.shape[0]):
#                   x = img[idx]
# #                   #print("Save Shape", x.shape, mask[idx].shape, torch.cat((x,x,x),1).shape)
#                   #img_list = [torch.cat((x,x,x),0),mask[idx],pred_mask[idx]]
#                   #utils.save_image(img_list,"../outputs11/image_outs/batch_out_{0}_{1}_{2}.png".format(epoch,batch_idx,idx))
# #  #                 print("Images saved to Directory")
# #               # print()
#               # epoch_train_loss.append(metrics["loss"]/len(train_loader.dataset))
# #  #             print("Testing on Validation Set")
# #               test_metrics = defaultdict(float)
# #               for test_idx,test_batch in enumerate(test_loader):
# #                   model.eval()
# #                   test_img  = test_batch[0].to(device)
# #                   test_mask = test_batch[1].to(device)
# #                   pred_test = model(test_img)
# #                   loss_test = loss(mask,pred_mask)
# #                   for idx in range(test_img.shape[0]):
# # #                       print("Save Shape", img[idx].shape, mask[idx].shape, pred_mask[idx].shape)
# #                        #img[idx] = cv2.resize(img[idx], (3,1,512,512), interpolation = cv2.INTER_AREA)
# #                        x = img[idx]
# #                        img_list = [torch.cat((x,x,x),0),mask[idx],pred_mask[idx]]
# #                        utils.save_image(img_list,"../outputs10/image_outs/test_batch_out_{0}_{1}_{2}.png".format(epoch,test_idx,idx))
# #   #                     print("Images saved to Test Directory")
# #               test_metrics["test_samples"] = len(test_loader.dataset)
# #               c = open("../outputs10/train_logs/test_metrics_{}_{}.json".format(batch_idx,epoch),'w+')
# #               with open("../outputs10/train_logs/test_metrics_{}_{}.json".format(batch_idx,epoch),'w') as wrf1:
# #                   json.dump(test_metrics, wrf1)
# #     b = open("../outputs10/train_logs/epoch_metrics_{}.json".format(epoch), 'w+')
# #     with open("../outputs10/train_logs/epoch_metrics_{}.json".format(epoch), 'w') as wrf2:
# #
# #
# #      json.dump(metrics, wrf2)
#     print("Epoch ", epoch, "loss: ", epoch_loss)
