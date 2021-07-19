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


EPS = 1e-10


def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.
    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.
    Args:
        hist: confusion matrix.
    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    std = torch.diag(hist)
    print(std)
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    std = torch.diag(hist)/(total + EPS)
    std = torch.std(std)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.
    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.
    Args:
        hist: confusion matrix.
    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    std_acc = nanstd(per_class_acc)
    return avg_per_class_acc,std_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
    Args:
        hist: confusion matrix.
    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    std_jacc = nanstd(jaccard)
    return avg_jacc,std_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    std_dice = nanstd(dice)
    return avg_dice,std_dice


def eval_metrics(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.
    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc,std_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc,std_jacc = jaccard_index(hist)
    avg_dice,std_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc,std_per_class_acc, avg_jacc,std_jacc, avg_dice,std_dice


def one_hot(target,num_classes):

    one_hot = torch.LongTensor(target.size(0) n,um_classes, target.size(1), target.size(2)).zero_()
    target = target.unsqueeze(1)
    target_one_hot = one_hot.scatter_(1, target.data.long(), 1)
    return target_one_hot

def save_image(prediction,batch_idx,num_classes):
    if prediction.shape[0] == 1:
        prediction = F.sigmoid(prediction)
        prediction = torch.round(prediction)
        torchvision.utils.save_image(prediction,"./infer_results/{}.png".format(batch_idx))
    else:
        prediction = F.softmax(prediction)
        _,indices = torch.argmax(prediction,0)
        indices = indices.unsqueeze(0)
        pred = one_hot(indices,num_classes)
        torchvision.utils.save_image(pred,"./infer_results/{}.png".format(batch_idx))



