import torch
import torch.nn as nn
from layers import *
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class ModifiedUNet(pl.LightningModule):

    def __init__(self,train_dataset=None,val_dataset=None, n_class=None,batch_size=None,lr=None,num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.sconv_down1 = single_conv(1, 64)
        self.sconv_down2 = single_conv(64, 128)
        self.sconv_down3 = single_conv(128, 256)
        self.sconv_down4 = single_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sconv_up3 = single_conv(256 + 512, 256)
        self.sconv_up2 = single_conv(128 + 256, 128)
        self.sconv_up1 = single_conv(128 + 64, 64)

        self.output_conv = nn.Conv2d(64,n_class,1)

        if n_class > 1:
            self.class_loss = nn.CrossEntropyLoss()
        else:
            self.class_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        conv1 = self.sconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.sconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.sconv_down3(x)
        x = self.maxpool(conv3)

        x = self.sconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.sconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.sconv_up2(x)
        x = self.upsample(x)

        x = torch.cat([x, conv1], dim=1)

        x = self.sconv_up1(x)

        out_single = self.conv_last_single(x)

        return out_single

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        if self.current_epoch < 16:
            loss = self.class_loss(y_hat,y)
        else:
            loss = self.class_loss(y_hat, y) + self.dice_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        if self.current_epoch < 16:
            v_loss = self.class_loss(y_hat,y)
        else:
            v_loss = self.class_loss(y_hat, y) + self.dice_loss(y_hat, y)
        return {'val_loss': v_loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_dataset, self.batch_size, self.batch_size)

    def dice_loss(target,input)
        smooth = 1.

        iflat = input.view(-1)
        iflat = torch.sigmoid(iflat)
        # print(torch.max(iflat))
        tflat = target.view(-1)
        # print(torch.max(tflat))
        intersection = (iflat * tflat).sum()
        
        return -1*torch.log((( intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth - intersection)))

