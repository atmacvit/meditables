import torch
import torch.nn as nn
import cv2
import torch.optim as optim
import time
import copy
from model import ModifiedUnet
import json
import torch.nn.functional as F
from data import *
import numpy as np
import argparse
import os
from torchvision import transforms,utils
from torch.utils.data import DataLoader
from pytorch_lightning import loggers,Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    #Training Arguments
    parser = argparse.ArgumentParser(description='For Getting Training Arguments')
    parser.add_argument('--train_dir',type=str,help='Path to Training Data')
    parser.add_argument('--val_dir',type=str,help="Path to Validation Data")
    parser.add_argument('--batch_size',type=int,default=4,help='Size of training batch')
    parser.add_argument('--num_class',type=int,default=1,help='Num of classes in Training Data')
    parser.add_argument('--num_workers',type=int,default=4,help="Number of workers for the DataLoader")
    parser.add_argument('--num_epoch',type=int,default=60,help='Number of epoch to Train the Model for')
    parser.add_argument('--lr',type=float,default = 0.001,help="Learning Rate for Training")
    parser = parser.parse_args()

    args = vars(parser)

    print('------------ Training Args -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('----------------------------------------')

    if args["num_class"]>1:
        binary = False
    else:
        binary =True



    train_transforms = torchvision.transforms.Compose([RandomNoise(),ToTensor()])
    validation_tranforms = torchvision.transforms.Compose([ToTensor()])



    print("Training on : {}".format(device))

    train_dataset = JsonDataset(args["train_dir"],transform = train_transforms,binary)
    val_dataset = JsonDataset(args["val_dir"],transform = validation_transforms,binary)



    if not os.path.exists("./outputs"):
        os.mkdir("output")

    tb_logger = loggers.TensorBoardLogger("./outputs")
    checkpoint_callback = ModelCheckpoint("./outputs/{epoch:02d}-{val_loss:.2f}.pth",save_top_k=2,mode='min')


    net = ModifiedUNet(train_dataset,val_dataset,args["num_class"],args["batch_size"],args["lr"],args["num_workers"])
    trainer = Trainer(checkpoint_callback=checkpoint_callback,gradient_clip_val=0.5,logger=tb_logger)
    trainer.fit(net)

