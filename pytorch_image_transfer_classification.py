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
import torchvision
from torchbearer import Trial
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToPILImage, ToTensor
from torchbearer.callbacks import TensorBoard
from torch.nn import Module
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import torch.optim as optim
from pathlib import Path
from pytorch_scratch_classification import Dataset
from torchvision import models, datasets
import copy
import time

if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Linear(in_features=2048, out_features=14, bias=True)
    train_prop = 0.8

    train_transformer = transforms.Compose([transforms.RandomRotation(40), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=False, train_proportion=train_prop)

    test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=True, train_proportion=train_prop)

    batch_size = 20
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    data_loader_dict = {'train': train_loader, 'eval': test_loader}
    optimizer =  optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_size = train_dataset.dataset_sub_size
    test_size = test_dataset.dataset_sub_size
    print(train_size)
    print(test_size)
    dataset_size = {'train': train_size, 'eval': test_size}



    'Model training and testing function'
    def train_model(model=model, optimizer=optimizer, loss_type = criterion, num_epochs = 3):
        best_model_weights = copy.deepcopy(model.state_dict()) #May be changed at end of each "for phase block"
        best_accuracy = 0 # May be changed at end of each "for phase block"
        start = time.time()
    
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
                        print('\n'*4)
                        print('#'*20)
                        print('Next input')
                        print(inputs)
                        print(inputs.shape)
                        print('#'*20)
                        print('Next output')
                        outputs = model(inputs)
                        print(outputs)
                        print(outputs.shape)
                        _, preds = torch.max(outputs, 1)
                        print(labels)
                        print(labels.shape)
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
        time_diff = time.time()-start
        print(f'Time taken for model to run: {(time_diff//60)} minutes and {(time_diff%60):.0f} seconds')
        return model

    model_tr = train_model()
    

    def show_image(input_ten_orig):
        input_ten = torch.clone(input_ten_orig)
        inv_normalize_array = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
        inv_normalize = transforms.Compose([inv_normalize_array])
        input_ten = inv_normalize(input_ten)
        plt.imshow(np.transpose(input_ten.numpy(), (1, 2, 0)))

    def images_to_proba(input_arr, model = model):
        output = model(input_arr)
        _, predicted_tensor = torch.max(output, 1)
        preds = np.squeeze(predicted_tensor.numpy())
        return preds, [F.softmax(out, dim=0)[pred_val].item() for pred_val, out in zip(preds, output)]

    def plot_classes_preds(input_arr, lab, model = model):
        preds, proba = images_to_proba(model, input_arr)
        fig = plt.figure(figsize=(12, 48))
        for i in range(4):
            ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
            show_image(input_arr[i])
        return fig
    # train_iterator = iter(train_loader)
    # img, label = train_iterator.next()
    # img_grid = torchvision.utils.make_grid(img)
    # writer = SummaryWriter()
    # writer.add_image('test_run', img_grid)
    
    # torchbearer_trial = Trial(model=model, optimizer=optimizer, criterion=criterion, metrics=['acc'], callbacks=[TensorBoard(write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])
    # torchbearer_trial.with_generators(train_generator=train_loader, val_generator=test_loader)
    # torchbearer_trial.run(epochs=1)

    

    


    
