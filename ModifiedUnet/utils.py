import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
import PIL
import cv2
from torchvision import transforms
from PIL import Image                                      # (pip install Pillow)# (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json




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

def preds_to_rgb_mask(prediction,num_classes):
    if num_classes == 1:
        prediction = F.sigmoid(prediction)
        prediction = torch.round(prediction)
        return transforms.ToPILImage()(prediction).convert('LA')
    else:
        prediction = F.softmax(prediction)
        _,indices = torch.argmax(prediction,0)
        indices = indices.unsqueeze(0)
        pred = one_hot(indices,num_classes)
        return transforms.ToPILImage()(prediction).convert('RGB')


def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation="low")

    polygons = []
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        
        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
    
    return polygons, segmentations

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

def create_json(mask_image_open,file_path,save_path):
    # This id will be automatically increased as we go
    multipolygon_ids = []
    category_ids = {
        "backgorund" : 0,
        "T1" : 1,
        "T2" : 2
     }

    category_colors = {
        "(0,0,0)" : 0,
        "(255,0,0)" : 1,
        "(0,255,0)":2

    }
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    
    w, h = mask_image_open.size
    image = create_image_annotation(file_path, w, h, image_id)
    images.append(image)
    sub_masks = create_sub_masks(mask_image_open, w, h)
    for color, sub_mask in sub_masks.items():
        category_id = category_colors[color]

        # "annotations" info
        polygons, segmentations = create_sub_mask_annotation(sub_mask)

        # Check if we have classes that are a multipolygon
        if category_id in multipolygon_ids:
            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)
                            
            annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

            annotations.append(annotation)
            annotation_id += 1
        else:
            for i in range(len(polygons)):
                # Cleaner to recalculate this variable
                segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                
                annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                
                annotations.append(annotation)
                annotation_id += 1
    image_id += 1
    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)
    coco_format["images"], coco_format["annotations"] = images,annotations
    # save_path = os.path.join(os.getcwd(),save_path)
    with open(save_path,"w") as f:
        json.dump(coco_format,f)







