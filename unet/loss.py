import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils import make_one_hot
from torch.autograd import Variable
from utils import dice_loss


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def loss_fn(pred, target,epoch):
    if epoch < 16:
        return -2*torch.log(dice_loss(pred, target))
    else:
        log_dice_loss = -2*torch.log(dice_loss)

        class_loss = F.cross_entropy(pred,target.long())
        return log_dice_loss + class_loss
