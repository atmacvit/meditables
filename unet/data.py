import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import PIL
from PIL import Image
import torch
import numpy
import numpy as np
import json


class UnetDataset(Dataset):
    def __init__(self,image_dir,labels_dir,transform=None,imageloader=cv2.imread):
        super(UnetDataset,self).__init__()
        self.images_dir = image_dir
        self.labels_dir = labels_dir
        self.images = [os.path.join(image_dir,x) for x in os.listdir(image_dir) ]
        self.labels = [os.path.join(labels_dir,x) for x in os.listdir(labels_dir) ]
        self.images.sort()
        self.labels.sort()
        self.transform = transform
        self.imageloader = imageloader

    def __len__(self):
        return len(self.images)


    def __getitem__(self,idx):
        image_path = self.images[idx]
        labels_path = self.labels[idx]
        img = self.imageloader(image_path)
        labels = self.imageloader(labels_path)
        if self.transform:
            img = self.transform(img)
            labels = self.transform(labels)
        return (img,labels)


class InferDataset(Dataset):
    def __init__(self,image_dir,transform,imageloader=cv2.imread):
        super(TestDataset,self).__init__()
        self.image_dir = image_dir
        self.images = [x for x in os.listdir(image_dir) if os.path.splitext(x)[1] in ['jpeg','png','jpg']].sort()
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














class JsonDataset(Dataset):
    def __init__(self,image_dir,labels_dir,tranform=None,imageloader=cv2.imread):
        super(JsonDataset,self).__init__()
        self.images_dir = image_dir
        self.labels_dir = labels_dir
        self.images = [x for x in os.listdir(image_dir) if os.path.splitext(x)[1] in ['jpeg','png','jpg']].sort()
        self.labels = [x for x in os.listdir(labels_dir) if x.endswith("json")].sort()

    def __len__(self):
        return len(self.images)

    def make_jsonimage(jsonfile,shape,img):
        canvas = img.copy()
        cv2.rectangle(canvas,(0,0),(shape[1],shape[0]),(0,0,0),-1)
        with open(jsonfile) as json_file:
            data = json.load(json_file)
            for each in data["shapes"]:
                print("ep", each["points"])
                a = np.array(each["points"])
                a = np.append(a, [a[0]], axis=0)
#                 print("a", a)
#                 a[:,0] = a[:,0]*512/img1.shape[0]
#                 a[:,1] = a[:,0]*512/img1.shape[1]
#                 print("a1", a)
                if each['label']=='Table':
                    rr, cc = polygon(a[:,0], a[:,1], img.shape)
                    canvas[cc,rr] = 1
                    print("Table Fig below")
#                     plt.figure(figsize=(8,8))
#                     plt.imshow(img)
#                     plt.show()
                if each['label']=='T1':
                    rr, cc = polygon(a[:,0], a[:,1], img.shape)
                    canvas[cc,rr] = 1
                    print("T1 Fig below")
#                     plt.figure(figsize=(8,8))
#                     plt.imshow(img)
#                     plt.show()
#                     cv2.fillPoly(img, [np.array(a, np.int64)], (0,1,0))
                if each['label']=='T2':
                    rr, cc = polygon(a[:,0], a[:,1], img.shape)
                    canvas[cc,rr] = 2
                    print("T2 Fig below")
                return canvas






    def __getitem__(self,idx):
        image_path = os.path.join(self.image_dir,self.images[idx])
        json_path = os.path.join(self.labels_dir,self.labels[idx])

        img = cv2.imread(image_path, 0)
        print("Actual Image Size", img.shape)
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        labelimg = make_jsonimage(json_path,img.shape,img)
        if self.transfrom:
             img = self.transform(img)
             labelimg = self.transform(labelimg)
        return (img,labelimg)
