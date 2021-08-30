# MediTables

This repository is for Meditables, a 200 image camera-captured dataset of medical reports with table annotations. Two tables types are common in healthcare documents which we have referred to as a T1 type table (conventional) and a T2 type table (key-value pairs). The dataset can be accesed here: https://zenodo.org/record/5048287#.YNzazBMzZhE

In this repo, we also provide pre-trained models in this repo for localizing these tables.

For additional details, check out our paper [MediTables: A New Dataset and Deep Network for Multi-Category Table Localization in Medical Documents](https://drive.google.com/file/d/1O1OI8Lc9xCuZolwUcWPTQcycWQGQ5piE/view?usp=sharing") accepted for ORAL presentation at <a href="https://grec2021.univ-lr.fr/">The 14th IAPR International Workshop on Graphics Recognition (GREC 2021)</a>.

Sample images with table annotations from MediTables dataset:
![Sample images with table annotations from MediTables dataset](https://user-images.githubusercontent.com/46661059/124064998-8137f180-da53-11eb-9ae5-e90bf96633e0.jpeg)


Comparison of our proposed model with baselines:
![Comparison of our proposed model with baselines](https://user-images.githubusercontent.com/46661059/124064982-7c733d80-da53-11eb-9e21-52b53993d3ba.png)

## Usage
All the model code is self contained in the Modified-Unet Directory,the baselines trained in the paper are described in the baselines directory. To run the code first we need to setup a virtualenv
To setup a Virtual Environment run :

```
pip install virtualenv 
virtualenv <environment name> --python=python3
source <environment name>/bin/activate
```

Then Install all the required packages
```
pip install -r requirements.txt
```

Once the packages installation is done , download the dataset from the above link.

The downloaded data is to be arranged in the following format:

```
.
+-- train
|   +-- 1.jpg
|   +-- 1.json
|   +-- ..
|   +-- ..
+-- Val
|   +-- 1.jpg
|   +-- 1.json
|   +-- ..
|   +-- ..
+-- test
|   +-- 1.jpg
|   +-- 1.json
|   +-- ..
|   +-- ..
```
i.e the the image files and the corresponding annotation files from a split are to be places in the same directory.

# To Train the model:
```
python ModifiedUnet/train.py --train_dir <path to directory containing training data > --val_dir <path to directory containing the validation data> --num_class <number of classes to train on>
```
Other parameters like **batch_size** and **lr** can also be modified.

Once the training is finished, the model checkpoints and tensoboard logs will be saved in `outputs` directory.

# To run inference on a trained model:
 ```
 python ModifiedUnet/inference.py --checkpoint_path <path to trained checkpoint> --infer_dir <Path to Inference Images Directory> --num_class <number of classes>
 ```
 Once the Inference is finished the predicted masks will be saved in the `infer_results` directory.
 
 




