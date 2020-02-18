import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Model_loss(nn.Module):
    def __init__():
        super(Model_loss, self).__init__()
    def log_dice(pred,target):
        pred_max = F.softmax(pred,dim =1)
        target_one_hot = one_hot(target,pred2.shape[1])
        dims = (1, 2, 3)
        intersection = torch.sum(pred_max * target_one_hot, dims)
        cardinality = torch.sum(pred_max + target_one_hot, dims)
        dice_score = (intersection+smooth) / (cardinality + smooth)
        n_log_dice = -1*torch.log(dice_score)
        return n_log_dice
    def forward(pred,target,epoch):
        if epoch<16:
            return log_dice(pred,target)
        else:
            return (F.cross_entropy(pred,target) + log_dice(pred_target))



      
