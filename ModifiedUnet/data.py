import cv2
import os
import PIL
from PIL import Image
import torch
import numpy as np
import json
import skimage
from torchvision import transforms,utils
from torch.utils.data import DataLoader,Dataset
from skimage.draw import polygon
import torchvision
import random

class JsonDataset(Dataset):
    """ Torch Dataset Class to load the annotations from the annotations directory"""
    def __init__(self,annotation_dir,transform=None,imageloader=cv2.imread,binary=True):
        super(JsonDataset,self).__init__()
        self.anno_dir = annotation_dir
        self.images = []
        self.labels = []
        self.binary = True
        self.transforms = transform
        for f in os.listdir(annotation_dir):
          if f.endswith(".json"):
            self.labels.append(f)
          else:
            self.images.append(f)
        self.images.sort()
        self.labels.sort()

    def __len__(self):
        return len(self.images)

    def make_jsonmask(self,jsonfile,shape,img):
        canvas = img.copy()
        cv2.rectangle(canvas,(0,0),(shape[1],shape[0]),(0,0,0),-1)
        with open(jsonfile) as json_file:
          data = json.load(json_file)
          for each in data["shapes"]:

              a = np.array(each["points"])
              a = np.append(a, [a[0]], axis=0)

              if each['label']=='Table':
                  rr, cc = polygon(a[:,0], a[:,1], img.shape)
                  canvas[cc,rr] = 1


              if each['label']=='T1':
                  rr, cc = polygon(a[:,0], a[:,1], img.shape)
                  canvas[cc,rr] = 1
 

              if each['label']=='T2':
                  rr, cc = polygon(a[:,0], a[:,1], img.shape)
                  if self.binary == True:
                    canvas[cc,rr] = 1
                  else:
                    canvas[cc,rr] = 2
              return canvas

    def __getitem__(self,idx):
        image_path = os.path.join(self.anno_dir,self.images[idx])
        json_path = os.path.join(self.anno_dir,self.labels[idx])

        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        labelimg = self.make_jsonmask(json_path,img.shape,img)
        sample = {"image":img,"labels":labelimg}
        if self.transforms:
             sample = self.transforms(sample)
        return (sample["image"],sample["labels"])

class InferDataset(Dataset):
    def __init__(self,image_dir,transform,imageloader=cv2.imread):
        super(TestDataset,self).__init__()
        self.image_dir = image_dir
        self.images = []
        for f in os.listdir(image_dir):
          self.images.append(f)
        self.transform = transform
        self.imageloader = imageloader

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image_path = self.images[idx]
        img = self.imageloader(image_path)
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        pil_image = PIL.Image.fromarray(image).convert('L')
        if self.transform:
            im_tensor = self.transform(pil_image)
        return im_tensor




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        image = np.expand_dims(image,axis = 0)
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

class RandomNoise(object):
  def __call__(self,sample):
    print(sample)
    gauss =  GuassianNoise()
    poiss = PoissonNoise()
    salt = SaltNoise()
    choice = random.choice([0,1,2,3])
    if choice == 0:
      return sample
    elif choice == 1:
      return gauss(sample)
    elif choice == 2:
      return poiss(sample)
    else:
      return salt(sample)
  



class GuassianNoise(object):
  """Apply Gaussian Noise to the Images at Random"""
  def __call__(self,sample):
    choice = random.choice([0,1])
    if choice == 0:
      return sample
    else:
      image,labels = sample['image'],sample['labels']
      gauss = np.random.normal(0,0.01,aimage.size)
      gauss = gauss.reshape(image.shape[0],image.shape[1]).astype('uint8')
      image = image + image * gauss
      return {'image': image,
                'labels': labels}

class PoissonNoise(object):
  """Apply Poisson Noise to the Images at Random"""
  def __call__(self,sample):
    choice = random.choice([0,1])
    if choice == 0:
      return sample
    else:
      image,labels = sample['image'],sample['labels']
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      image = np.random.poisson(np.abs(image * vals)) / float(np.abs(vals))
      return {'image': image,
                'labels': labels}
class SaltNoise(object):
  """Apply Salt Noise to the Images at Random"""
  def __call__(self,sample):
    choice = random.choice([0,1])
    if choice == 0:
      return sample
    else:
      image,labels = sample['image'],sample['labels']
      noise_img = skimage.util.random_noise(image, mode='salt',amount=0.12) 
      image = np.array(255*noise_img, dtype = 'uint8')
      return {'image': image,
                'labels': labels}

class RandomAffine(object):
  """Apply a Random Affine Transform to the Data"""
  def __call__(self,sample):
    choice = random.choice([0,1])
    if choice == 0:
      return sample
    else:
      image,labels = sample['image'],sample['labels']
      k = np.random.normal(2,0.4)
      j = np.random.normal(0,10)
      M = cv2.getRotationMatrix2D((image.shape[1],image.shape[0]),j,k)
      image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]), borderMode=cv2.BORDER_REFLECT)
      labels = cv2.warpAffine(labels,M,(labels.shape[1],labels.shape[0]), borderMode=cv2.BORDER_REFLECT)
      return {'image': image,
                'labels': labels}