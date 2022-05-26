from turtle import color
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
from torch.optim import lr_scheduler
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
    res_model = models.resnet50(pretrained=True)
    for param in res_model.parameters():
        param.requires_grad = False
    
    opt = optim.SGD
    res_model.fc = nn.Linear(in_features=2048, out_features=13, bias=True)
    train_prop = 0.8

    train_transformer = transforms.Compose([transforms.RandomRotation(40), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=False, train_proportion=train_prop)
    classes = list(train_dataset.classes)
    class_codes = train_dataset.encoded_class
    class_encoder = train_dataset.class_dict

    test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=True, train_proportion=train_prop)

    # test_classes = train_dataset.classes
    # test_class_codes = train_dataset.encoded_class
    # test_class_encoder = train_dataset.class_dict


    batch_size = 32
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    data_loader_dict = {'train': train_loader, 'eval': test_loader}
    optimizer =  opt(res_model.parameters(), lr=0.1)
    # lambda_scheduler = lambda epoch: epoch*0.8 if epoch<=16  else epoch*0.1
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_scheduler)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4, 8, 12, 15, 20, 22], gamma=0.5) 
    criterion = nn.CrossEntropyLoss()

    train_size = train_dataset.dataset_size
    test_size = test_dataset.dataset_size
    print(train_size)
    print(test_size)
    dataset_size = {'train': train_size, 'eval': test_size}

    # print(train_classes == test_classes)
    # print(train_class_codes == test_class_codes)
    # print(train_class_encoder == test_class_encoder)
    # print(train_classes)
    # print(train_class_codes)
    # print(train_class_encoder)

    writer = SummaryWriter()

    '''Tensorboard Function for Showing Images'''
    def show_image(input_ten_orig):
        input_ten = torch.clone(input_ten_orig)
        inv_normalize_array = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
        inv_normalize = transforms.Compose([inv_normalize_array])
        input_ten = inv_normalize(input_ten)
        input_numpy = input_ten.numpy()
        plt.imshow(np.transpose(input_numpy, (1, 2, 0)))
        # plt.show()

    '''Function for comparing actual images to predicted images in Tensorboard'''
    def images_to_proba(input_arr, model = res_model): #Stub function used in plot_classes_preds to 
        input_tensor = torch.clone(input_arr)
        output = model(input_tensor)
        _, predicted_tensor = torch.max(output, 1)
        preds = np.squeeze(predicted_tensor.numpy())
        return preds, [F.softmax(out, dim=0)[pred_val].item() for pred_val, out in zip(preds, output)]

    def plot_classes_preds(input_arr, lab, model = res_model):
        preds, proba = images_to_proba(input_arr, model)
        print(preds)
        print(proba)
        fig = plt.figure(figsize=(12, 12))
        for i in range(4):
            ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
            show_image(input_arr[i])
            ax.set_title('{0}, {1:.1f}%\n(label: {2})'.format(classes[preds[i]], proba[i]*100, classes[lab[i]]), color=('green' if preds[i]==lab[i].item() else 'red')) #
            plt.tight_layout()
        return fig



    'Model training and testing function'
    def train_model(model=res_model, optimizer=optimizer, loss_type = criterion, num_epochs = 30, mode_scheduler = scheduler):
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

                for batch_num, (inputs, labels) in enumerate(data_loader_dict[phase], start=1):
                    optimizer.zero_grad() # Gradients reset to zero at beginning of both training and evaluation phase

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        outputs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)
                        loss = loss_type(outputs, labels)
                        if phase == 'train':
                            loss.backward() #Calculates gradients
                            optimizer.step()
                                # img_grid = torchvision.utils.make_grid(inputs)
                                # writer.add_image('Image of product', show_image(img_grid))
                                # writer.add_figure('Predictions vs Labels', plot_classes_preds(model, input_arr=inputs, lab=labels))

                    if batch_num%10==0:
                        '''Writer functions for batch'''
                        writer.add_figure('Predictions vs Actual',plot_classes_preds(input_arr=inputs, lab=labels, model=model))
                        writer.add_scalar(f'Accuracy for phase {phase} by batch number', preds.eq(labels).sum()/batch_size, batch_num)
                        writer.add_scalar(f'Average loss for phase {phase} by batch number', loss.item(), batch_num)

                    running_corrects = running_corrects + preds.eq(labels).sum()
                    running_loss = running_loss + (loss.item()*inputs.size(0))

                if phase=='train':
                    mode_scheduler.step()

                '''Writer functions for epoch'''
                epoch_loss = running_loss / dataset_size[phase]
                print(f'Size of dataset for phase {phase}', dataset_size[phase])
                epoch_acc = running_corrects / dataset_size[phase]
                writer.add_scalar(f'Accuracy by epoch phase {phase}', epoch_acc, epoch)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                writer.add_scalar(f'Average loss by epoch phase {phase}', epoch_loss, epoch)

                if phase == 'eval' and epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print(f'Best val Acc: {best_accuracy:.4f}')


        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), 'image_model.pt')
        time_diff = time.time()-start
        print(f'Time taken for model to run: {(time_diff//60)} minutes and {(time_diff%60):.0f} seconds')
        return model

    model_tr = train_model()





    # train_iterator = iter(train_loader)
    # img, label = train_iterator.next()
    # img_grid = torchvision.utils.make_grid(img)
    # writer = SummaryWriter()
    # writer.add_image('test_run', img_grid)
    
    # torchbearer_trial = Trial(model=model, optimizer=optimizer, criterion=criterion, metrics=['acc'], callbacks=[TensorBoard(write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])
    # torchbearer_trial.with_generators(train_generator=train_loader, val_generator=test_loader)
    # torchbearer_trial.run(epochs=1)

    

    


    
