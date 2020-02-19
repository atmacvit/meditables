import torch
import torch.nn as nn
import numpy as np
import os
import numpy as np
import torch.nn.functional as F
import PIL
import cv2
from torch.autograd import Variable
from torch.nn.init import xavier_uniform


def tot(pred, target):
    return -2*torch.log(dice_loss(pred, target)) + calc_loss(pred, target)[0]

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def load_train_test(dataset,valid_size = 0.001):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
#    print(split)
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
   # train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(indices)
    #test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(dataset,
                   sampler=train_sampler, batch_size=8,pin_memory=True)
    #testloader = torch.utils.data.DataLoader(dataset,
     #              sampler=test_sampler, batch_size=1,pin_memory=False)
    return trainloader #testloader

def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

#    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
#    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
#    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return bce, loss

def one_hot(target,num_classes):
    # a, b, height, width = target.shape
    #one_hot = torch.zeros(a, num_classes, height, width)
    one_hot = torch.cuda.LongTensor(target.size(0), num_classes, target.size(1), target.size(2)).zero_()
#    print(one_hot.shape)
    #c = torch.tensor([1]).cuda()
    target = target.unsqueeze(1)
#    print(target.shape)
    target_one_hot = one_hot.scatter_(1, target.data.long(), 1)
#    print(torch.unique(target_one_hot))
    return target_one_hot

def loss(target1,target2,pred1,pred2,class_weights = None,multiclass = False):
    smooth = 1.
    # if multiclass:
    pred_max_2 = F.softmax(pred2,dim =1)
    target_one_hot_2 = one_hot(target2,pred2.shape[1])
    dims = (1, 2, 3)
    intersection = torch.sum(pred_max_2 * target_one_hot_2, dims)
    cardinality = torch.sum(pred_max_2 + target_one_hot_2, dims)
    dice_score = (intersection+smooth) / (cardinality + smooth)
    n_log_dice = -1*torch.log(dice_score)
    cce_loss = F.cross_entropy(pred2,target2.long())
    l1 = (cce_loss + 2*(n_log_dice))
    # else:
   #  dims = (1, 2, 3)
    pred_max_1 = torch.sigmoid(pred1)
    # print(target1.shape)
    # print(pred1.shape[1])
    # target_one_hot_1 = one_hot(target1,pred1.shape[1])
    intersection = torch.sum(pred_max_1 * target1,dims)
    cardinality = torch.sum(pred_max_1 + target1,dims)
    s_dice_score =  (intersection+smooth) / (cardinality + smooth)
    n_log_dice1 = -1*torch.log(dice_score)
    #    print(n_log_dice.shape)
    bce_loss = F.binary_cross_entropy_with_logits(pred1,target1.float())
     #   print(bce_loss.shape)
   #     print("Log Loss : {} || BCE Loss : {}".format(n_log_dice,bce_loss))
    l2 = (bce_loss + 2*(n_log_dice1))
    l = l1 + l2
    return l.mean()


def make_one_hot(labels):

    one_hot = torch.cuda.FloatTensor(labels.size(0), labels.size(1), labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target

def load_model(model,checkpoint):
    pretrained_dict = torch.load(checkpoint)
    model_dict = model.state_dict()
    # print("M", model_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def save_image(prediction,batch_idx):
    if prediction.shape[0] == 1:
        prediction = F.sigmoid(prediction)
        prediction = torch.round(prediction)
        torchvision.utils.save_image(prediction,"./infer_results/{}.png".format(batch_idx))
    else:
        prediction = F.softmax(prediction)
        _,indices = torch.argmax(prediction,0)
        indices = indices.unsqueeze(0)
        pred = one_hot(indices)
        torchvision.utils.save_image(pred,"./infer_results/{}.png".format(batch_idx))


def weights_init(m):
   if isinstance(m, nn.Conv2d):
       xavier_uniform(m.weight.data)
       xavier_uniform(m.bias.data)
