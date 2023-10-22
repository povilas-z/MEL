import torchvision
from torchvision import transforms
import os
import shutil
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.utils import resample
import pandas as pd
import random

#DATA RESTRUCTURING
# data_dir = os.getcwd() + "/HAM10000/all_images/"
# dest_dir = os.getcwd() + "/HAM10000/reorganized/"

# skin_df = pd.read_csv('HAM10000/HAM10000_metadata.csv')
# print(skin_df['dx'].value_counts())

# labels = skin_df['dx'].unique().tolist()
# label_images = []

# for label in labels:
#     os.mkdir(dest_dir + str(label) + "/")
#     sample = skin_df[skin_df['dx'] == label]['image_id']
#     label_images.extend(sample)
#     for id in label_images:
#         shutil.move((data_dir + "/" + id + ".jpg"), (dest_dir + label + "/" + id))
#     label_images = []


#PYTORCH TENSOR CREATION
train_dir = os.getcwd() + "/HAM10000/reorganized/"

def getTransforms():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.229,0.224,0.225])
    ])

def getDataSet(root_dir, transforms):
    return torchvision.datasets.ImageFolder(root = root_dir, transform = transforms)

def getDataLoader(dataset, batch_size = 32, shuffle = True):
    return DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)