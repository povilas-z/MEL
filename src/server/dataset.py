import torchvision
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Subset
import random
import shutil
import pandas as pd

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


class DataSet:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, "reorganized/")
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225])
        ])
    
    def resampleDataSet(self, targetSamples):
        full_dataset = torchvision.datasets.ImageFolder(root=self.train_dir, transform=self.transform)
        class_indices = self.getClassIndices(full_dataset)
        balanced_indices = []
        for indices in class_indices.values():
            if len(indices) > targetSamples:
                # Downsample
                balanced_indices.extend(random.sample(indices, targetSamples))
            else:
                # Oversample
                balanced_indices.extend(indices + [random.choice(indices) for _ in range(targetSamples - len(indices))])
        # Create a dataset
        balanced_subset = Subset(full_dataset, balanced_indices)
        return balanced_subset
    
    def getClassIndices(self, dataset):
        class_indices = {}
        for idx, (_, target) in enumerate(dataset):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)
        return class_indices
    
    def prepareDataSet(self, targetSamples):
        balanced_dataset = self.resampleDataSet(targetSamples)
        return balanced_dataset
    
    @staticmethod
    def getDataLoader(dataset, batch_size=32, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)