import torch
from torch.nn.init import xavier_uniform
import torch.nn as nn
import cv2
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from networks import UNet
#from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from data import SegDataset, SegDataset1
import json
from collections import defaultdict
import torch.nn.functional as F
from utils import dice_loss,load_train_test,calc_loss,one_hot,loss,tot
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

num_class = 3
num_epochs = 1200

#image_dirs = ["../Datasets_and_Masked/ICDAR_2013_Actual","../Datasets_and_Masked/ICDAR_2017_Actual","../Datasets_and_Masked/Marmot_Actual","../Datasets_and_Masked/UNLV_Actual","../Datasets_and_Masked/Train_Actual_Images"]
#mask_dirs = ["../Datasets_and_Masked/ICDAR_2013_Masked","../Datasets_and_Masked/ICDAR_2017_Masked","../Datasets_and_Masked/Marmot_Masked","../Datasets_and_Masked/UNLV_Masked","../Datasets_and_Masked/Train_Masked_Images"]

#image_dirs = ["../../pixlabel/datasets/cityscapes/train_A"]
#mask_dirs = ["../../pixlabel/datasets/cityscapes/train_B"]

# image_dirs = ["../MedTrain_Actual", "../MedVal_Actual"]
# mask_dirs = ["../MedTrain_Masked", "../MedVal_Masked"]
#
test_dirs = ["../MedTest_Actual"]
mask_dirs1 = ["../MedTest_Masked"]

train_transforms = transforms.Compose([
    transforms.ToTensor()])

data = SegDataset("train_data",transform = train_transforms)
#print("THe size of the complete dataset is {}".format(data.__len__()))

train_loader,test_loader = load_train_test(data,valid_size = 0.01)

#print("The size of training examples is : {}".format(len(train_loader.dataset)))
#print("The size of testing examples is : {}".format(len(test_loader.dataset)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet(num_class).to(device)
#summary(model,(1,512,512))
# d = open('sample.json', 'w+')

optimizer = optim.Adam(model.parameters(), lr=0.0001)

#def weights_init(m):
#    if isinstance(m, nn.Conv2d):
#        xavier_uniform(m.weight.data)
#        xavier_uniform(m.bias.data)

#model.apply(weights_init)

model.load_state_dict(torch.load("../outputs12/checkpoints/ckpt_0_0.pth"))
transforms1 = transforms.Compose([
    transforms.ToTensor()])

#print(torch.load("../outputs4/checkpoints/ckpt_0_160.pth").keys())
#pretrained_dict = torch.load("../outputs4/checkpoints/ckpt_0_160.pth")
#model_dict = model.state_dict()
#print("M", model_dict.keys())
#for name,param in pretrained_dict.items():
#    if name not in model_dict:
#       continue
#    if isinstance(param, torch.nn.Parameter):
#                # backwards compatibility for serialized parameters
#        print("P loaded")
#        param = param.data
#        own_state[name].copy_(param)
#pretrained_dict = {k: v for k, v in pretrained_dict.items() if
#                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
#model_dict.update(pretrained_dict)
#model.load_state_dict(model_dict)
#key_list = [k for k in pretrained_dict.keys() for k in model_dict]
#print(key_list)


test_data = SegDataset1(test_dirs[0],transform=transforms1)
t_dataloader = DataLoader(test_data,batch_size=1,shuffle=False,pin_memory=False)
print(len(test_data))

count1 = 0
for t_data in t_dataloader:
    count1 = count1 + 1
    tr_data = Variable(t_data[0].cuda())
    print("File", t_data[1][0][18:])
    print(t_data[0].shape)
    t_gen = model(tr_data)
    print(t_gen)
    gen_out = Variable(tr_data).detach().cpu()
    utils.save_image(gen_out, '../outputs12/testresults/A'+t_data[1][0][18:]+'.jpg')
    gen_out = t_gen[0].detach().cpu()
    utils.save_image(gen_out, '../outputs12/testresults/G'+t_data[1][0][18:]+'.jpg')
    gen_out = t_gen[1].detach().cpu()
    utils.save_image(gen_out, '../outputs12/testresults/D'+t_data[1][0][18:]+'.jpg')
    print("Saved Image")

#pretrained_dict = torch.load("../outputs4/checkpoints/ckpt_0_180.pth")
#model_dict = model.state_dict()
#pretrained_dict = {k:v for k,v in pretrained_dict.items() for k in model_dict}
#model_dict.update(pretrained_dict)
#model.load_state_dict(pretrained_dict)

model = nn.DataParallel(model).to(device)
# model = model.to(device)
# criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    metrics = defaultdict(float)
    epoch_loss = 0
    train_loss = []
    # val_loss = []
    # epoch_train_loss = []
    # epoch_val_loss = []
    epoch_samples = 0
#    print("Epoch Loss for Epoch : {} is {} ".format(epoch,epoch_loss))
    for batch_idx,batch in enumerate(train_loader):
          model.train()
          img = batch[0].to(device)
          mask1 = batch[1].to(device)
          mask2 = batch[2].to(device)
# #          print("Shapes", img.shape, mask.shape)
          epoch_samples += img.size(0)
          pred_mask1,pred_mask2 = model(img)
          loss_model = loss(mask1,mask2,pred_mask1,pred_mask2)
#          print(loss_model.item())
#           #loss_model = criterion(pred_mask, mask)
          loss_model = loss_model.mean()
          epoch_loss += loss_model.item()
          optimizer.zero_grad()
          loss_model.backward()
          optimizer.step()
# #          print("EPOCH:{0} || BATCH NO:{1} || LOSS:{2}".format(epoch,batch_idx,loss_model.item()))
          if batch_idx%3000 == 0:
              torch.save(model.module.state_dict(),"../outputs12/checkpoints/ckpt_{}_{}.pth".format(batch_idx,epoch))
              metrics["batch_idx"] = batch_idx
              metrics["epoch"] = epoch
              metrics["epoch_samples"] = epoch_samples
              a = open("../outputs12/train_logs/train_metrics_{}_{}.json".format(batch_idx,epoch), 'w+')
              with open("../outputs12/train_logs/train_metrics_{}_{}.json".format(batch_idx,epoch), 'w') as wrf:
                  json.dump(metrics, wrf)
              for idx in range(img.shape[0]):
                  x = img[idx]
#                   #print("Save Shape", x.shape, mask[idx].shape, torch.cat((x,x,x),1).shape)
                  #img_list = [torch.cat((x,x,x),0),mask[idx],pred_mask[idx]]
                  #utils.save_image(img_list,"../outputs11/image_outs/batch_out_{0}_{1}_{2}.png".format(epoch,batch_idx,idx))
#  #                 print("Images saved to Directory")
#               # print()
              # epoch_train_loss.append(metrics["loss"]/len(train_loader.dataset))
#  #             print("Testing on Validation Set")
#               test_metrics = defaultdict(float)
#               for test_idx,test_batch in enumerate(test_loader):
#                   model.eval()
#                   test_img  = test_batch[0].to(device)
#                   test_mask = test_batch[1].to(device)
#                   pred_test = model(test_img)
#                   loss_test = loss(mask,pred_mask)
#                   for idx in range(test_img.shape[0]):
# #                       print("Save Shape", img[idx].shape, mask[idx].shape, pred_mask[idx].shape)
#                        #img[idx] = cv2.resize(img[idx], (3,1,512,512), interpolation = cv2.INTER_AREA)
#                        x = img[idx]
#                        img_list = [torch.cat((x,x,x),0),mask[idx],pred_mask[idx]]
#                        utils.save_image(img_list,"../outputs10/image_outs/test_batch_out_{0}_{1}_{2}.png".format(epoch,test_idx,idx))
#   #                     print("Images saved to Test Directory")
#               test_metrics["test_samples"] = len(test_loader.dataset)
#               c = open("../outputs10/train_logs/test_metrics_{}_{}.json".format(batch_idx,epoch),'w+')
#               with open("../outputs10/train_logs/test_metrics_{}_{}.json".format(batch_idx,epoch),'w') as wrf1:
#                   json.dump(test_metrics, wrf1)
#     b = open("../outputs10/train_logs/epoch_metrics_{}.json".format(epoch), 'w+')
#     with open("../outputs10/train_logs/epoch_metrics_{}.json".format(epoch), 'w') as wrf2:
#
#
#      json.dump(metrics, wrf2)
    print("Epoch ", epoch, "loss: ", epoch_loss)
