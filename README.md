# MediTables-IIIT

The Dataset, Code and Pre-trained Models for table localization in document images from the healthcare/medical domain 

We have collected a 200 image camera-captured dataset from the healthcare/medical domain that structurizes tables in a distinct manner compared to traditional documents. Two tables types are common in healthcare documents which we have referred to as a T1 type table (conventional) and a T2 type table (key-value pairs). 

The dataset MediTables-IIIT has been published in this repository along with the annotations for table types T1 and T2.

Our goal to locate tables has been carried out in two phases in order to develop baselines on the contributed dataset:
- Table Detection - Building of a semantic segmentation Modified-UNet model that outputs a target map with two classes - Table and Non-Table
- Table Segmentation - Building of a semantic segmentation Modified-UNet model that outputs a target map with three classes - Table T1, Table T2 and Non-Table

In order to evaluate the accuracy of our model in terms of three metrics: 
Intersection over Union, Per-Pixel accuracy and F1 score,
We have conducted experiments on three other models: 
1. An object recognition model TableBank - https://github.com/doc-analysis/TableBank
2. An object recognition model YOLOv3
3. A semantic segmentation model pix2pixHD - https://github.com/NVIDIA/pix2pixHD

Five popular datasets - Marmot, UNLV, ICDAR 2013 table competition, UW3 and TableBank have been used to train these models beforehand, before finetuning the models on MediTables-IIIT.
These datasets were pre-processed and augmented (code available in this repository).

Table Detection using Modified U-Net:
- Training the model using the five popular datasets - Model M1
- The training of the model using the training set of MediTables-IIIT was carried out by optimizing the model over two losses: per-pixel cross entropy loss & logarithmic version of IoU loss (only epoch 16 onwards with a coefficient of 20) for 58 epochs - This was evaluated on the validation set of MediTables-IIIT
- Using the same hyperparameters and the developed stopping criterion, Model M1 was trained using training and validation sets of MediTables-IIIT and evaluated one time on the testing set of MediTables-IIIT

Post-processing was also performed for all models before evaluation (code in this repository).
