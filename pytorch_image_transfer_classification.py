import pandas as pd
from clean_images import CleanImages
from clean_tabular import CleanData
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import re
from PIL import Image
import multiprocessing
from torchvision.transforms import Normalize, ToPILImage, ToTensor
from torch.nn import Module
from torch import nn
import torch.optim as optim
from pathlib import Path
from pytorch_scratch_classification import Dataset
from torchvision import models, datasets
import copy

if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(in_features=2048, out_features=13, bias=True)

    train_transformer = transforms.Compose([transforms.RandomRotation(40), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=False)

    test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=True)

    batch_size = 20
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    data_loader_dict = {'train': train_loader, 'eval': test_loader}
    optimizer =  optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_size = 10000
    test_size = 2424
    dataset_size = {'train': train_size, 'eval': test_size}



    'Model training and testing function'
    def train_model(model=model, optimizer=optimizer, loss_type = criterion, num_epochs = 20):
        best_model_weights = copy.deepcopy(model.state_dict()) #May be changed at end of each "for phase block"
        best_accuracy = 0 # May be changed at end of each "for phase block"

        for epoch in range(num_epochs):
            for phase in ['train', 'eval']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0
                running_corrects = 0

                for inputs, labels in data_loader_dict[phase]:
                    optimizer.zero_grad() # Gradients reset to zero at beginning of both training and evaluation phase

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_type(outputs, labels)

                        if phase == 'train':
                            loss.backward() #Calculates gradients
                            optimizer.step()
                    
                    running_loss = running_loss + loss.item()*inputs.size(0)
                    running_corrects = running_corrects + torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_size[phase]
                epoch_acc = running_corrects.double() / dataset_size[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'eval' and epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print(f'Best val Acc: {best_accuracy:.4f}')
        model.load_state_dict(best_model_weights)
        return model

    model_tr = train_model()