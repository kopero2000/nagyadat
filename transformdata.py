
import bamboolib as bm
import pandas as pd
import plotly.express as px
from skimage import io
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.io
import datetime
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.io import read_image
metadata = scipy.io.loadmat("/mnt/idms/home/a100/vizibela/data/imdb_crop/imdb.mat")
labels = []
for i in range(0, len(metadata['imdb']['dob'][0][0][0])):
    bd = datetime.datetime.fromordinal(metadata['imdb']['dob'][0][0][0][i]).year
    labels.append([metadata["imdb"]['full_path'][0][0][0][i][0], metadata['imdb']['photo_taken'][0][0][0][i] - bd])# one hot ?

#labels = labels[labels[1] < 60]
labels = filter(lambda c: c[1] > 11 and c[1] < 50, labels)
labels = list(labels)



from PIL import Image
import os
from torch.utils.data import Dataset
import PIL
import os.path
from os import path

count = 0

class ImageDataset(Dataset):
    def __init__(self, dir, start = 0, end = 10 ):
        self.img_labels = labels[start:end]
        self.dir = dir
        self.images = []
        i = 0
        for im in self.img_labels:

            i = i+ 1
            imgpath = "/mnt/idms/home/a100/vizibela/data/imdb_crop/"+im[0]
            if path.exists(imgpath):
                image = PIL_image = PIL.Image.open(imgpath).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                self.images.append(transform(image)) #itt resize + totensor
        self.images = torch.stack(self.images)
        transform = transforms.Compose([])

    def settransform(self, transform):
        self.transform = transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        return img, self.img_labels[idx][1]

    
    def read_data(size, validation_size, test_size):
    test_start = int(size-(test_size*size))
    valid_start = int(test_start-(validation_size*size))
    print(size, test_start, valid_start)
    data_train = ImageDataset('imdb', 0, valid_start)
    data_valid = ImageDataset('imdb', valid_start, test_start)
    data_test = ImageDataset('imdb', test_start, size)
    return data_train, data_valid, data_test

data_train, data_valid, data_test = read_data(100000, 0.125, 0.125)

data_train.settransform(transforms.Compose([]))
data_valid.settransform(transforms.Compose([]))
data_test.settransform(transforms.Compose([]))

dataloader_train_notmean = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True)
dataloader_v_notmean = torch.utils.data.DataLoader(data_valid, batch_size=32, shuffle=True)
dataloader_test_notmean = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=True)


def mean_std(loader):
  _images, lebels = next(iter(loader))
  # shape of images = [b,c,w,h]
  _mean, _std = _images.mean([0,2,3]), _images.std([0,2,3])
  return _mean, _std
_mean, _std = mean_std(dataloader_train_notmean)
print(_mean,_std)
transform =   transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.Normalize(_mean, _std)

])
data_train.settransform(transform)

_mean, _std = mean_std(dataloader_v_notmean)
print(_mean,_std)
transform =   transforms.Compose([

        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.Normalize(_mean, _std)
])
data_valid.settransform(transform)

_mean, _std = mean_std(dataloader_test_notmean)
print(_mean,_std)
transform =   transforms.Compose([

        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.Normalize(_mean, _std)

])
data_test.settransform(transform)


dataloader_train_normal = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True)
dataloader_v_normal = torch.utils.data.DataLoader(data_valid, batch_size=32, shuffle=True)
dataloader_test_normal = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=True)


torch.save(dataloader_train_normal, "dataloader_train_normal.pth")
torch.save(dataloader_v_normal, "dataloader_v_normal.pth")
torch.save(dataloader_test_normal, "dataloader_test_normal.pth")
torch.save(data_train, 'data_train.pt')
torch.save(data_valid, 'data_valid.pt')
torch.save(data_test, 'data_test.pt')
