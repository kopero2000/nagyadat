import bamboolib as bm
import pandas as pd
import plotly.express as px
from skimage import io
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
import datetime
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.io import read_image
from PIL import Image
import os
from torch.utils.data import Dataset
import PIL
import os.path
from os import path
import torchvision.models as models
from torch import nn
import copy


class ImageDataset(Dataset):
    def __init__(self, dir, start = 0, end = 10 ):
        self.img_labels = labels[start:end]
        self.dir = dir
        self.images = []

        for im in self.img_labels:
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

data_train = torch.load('data_train.pt')
data_valid = torch.load('data_valid.pt')
data_test = torch.load('data_test.pt')

dataloader_train_normal = torch.load('dataloader_train_normal.pth')
dataloader_v_normal = torch.load('dataloader_v_normal.pth')
dataloader_test_normal = torch.load('dataloader_test_normal.pth')


device = torch.device("cuda:5") #cude:5
print(torch.cuda)
model_ft = models.vgg16(pretrained=True).to(device)
model_ft.classifier[4] = nn.Linear(4096,1024)
model_ft.classifier[6] = nn.Linear(1024,40)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)


def train_model(model, criterion, optimizer, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                loader = dataloader_train_normal
            else:
                loader = dataloader_v_normal
            # Iterate over data.
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1)

                    labels = labels -11
                    loss = criterion(outputs, labels.long())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(data_train)
            epoch_acc = running_corrects.double() / len(data_train)
            if best_acc < epoch_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
        print()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
model_ft = train_model(model_ft, criterion, optimizer_ft)
torch.save(model_ft, 'model.pth')
