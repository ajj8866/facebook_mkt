from clean_tabular import CleanData, CleanImages, MergedData
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import re
import multiprocessing
import torchvision
from torchbearer import Trial
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torchbearer.callbacks import TensorBoard
from torch.nn import Module
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import torch.optim as optim
from pytorch_scratch_classification import Dataset
from torchvision import models, datasets
import copy
import time
import pandas as pd
import numpy as np

if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 400)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    plt.rc('axes', titlesize=12)

    res_model = models.resnet50(pretrained=True)
    for i, param in enumerate(res_model.parameters(), start=1):
        if i <=48:
            param.requires_grad = False
        else:
            param.requires_grad = True

    res_model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(in_features=512, out_features=64), nn.Linear(in_features=64, out_features=13))


    opt = optim.SGD
    optimizer =  opt(res_model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5, 10, 15, 20, 25, 30], gamma=0.3) 
    criterion = nn.CrossEntropyLoss()


    def get_loader(img = 'image_array',batch_size=35, split_in_dataset = True, train_prop = 0.8):
        train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(40), transforms.RandomGrayscale(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if split_in_dataset == True:
            train_dataset = Dataset(transformer=train_transformer, X=img, img_size=224, is_test=False, train_proportion=train_prop)
            test_dataset = Dataset(transformer=test_transformer, X=img, img_size=224, is_test=True, train_proportion=train_prop)
            dataset_dict = {'train': train_dataset, 'eval': test_dataset}
            data_loader_dict = {i: DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']}
            return train_dataset.dataset_size, test_dataset.dataset_size, data_loader_dict
        else:
            imaage_datsets= Dataset(transformer=test_transformer, X = img, img_size=224, is_test=None)
            train_end = int(train_prop*imaage_datsets.dataset_size)
            train_dataset, test_dataset = random_split(imaage_datsets, lengths=[len(imaage_datsets.all_data.iloc[:train_end]), len(imaage_datsets.all_data.iloc[train_end:])])
            dataset_dict = {'train': train_dataset, 'eval': test_dataset}
            data_loader_dict = {i: DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']}
            return len(imaage_datsets.all_data.iloc[:train_end]), len(imaage_datsets.all_data.iloc[train_end:]), data_loader_dict
        
    prod_dum = CleanData()
    class_dict = prod_dum.major_map_encoder.keys()
    classes = list(class_dict)
    class_values = prod_dum.major_map_encoder.values()
    class_encoder = prod_dum.major_map_encoder

######################################################################################################################################################################################
###################################     GRAPHS FOR TENSORBOARD #######################################################################################################################
######################################################################################################################################################################################

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

######################################################################################################################################################################################
###################################     MODEL TRAINING FUNCTION ######################################################################################################################
######################################################################################################################################################################################


    'Model training and testing function'
    def train_model(model=res_model, optimizer=optimizer, loss_type = criterion, num_epochs = 25, mode_scheduler=None, batch_size = 24, image_type='image_array', split_in_datset=False):
        best_model_weights = copy.deepcopy(model.state_dict()) #May be changed at end of each "for phase block"
        best_accuracy = 0 # May be changed at end of each "for phase block"
        start = time.time()
        writer = SummaryWriter()
        train_size, test_size, data_loader_dict = get_loader(batch_size=batch_size, img=image_type, split_in_dataset=split_in_datset)
        dataset_size = {'train': train_size, 'eval': test_size}
    
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
                        # print(inputs)
                        # print(inputs.size())
                        outputs = model(inputs)
                        #outputs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)
                        loss = loss_type(outputs, labels)
                        if phase == 'train':
                            loss.backward() #Calculates gradients
                            optimizer.step()

                    if batch_num%20==0:
                        '''Writer functions for batch'''
                        writer.add_figure('Predictions vs Actual',plot_classes_preds(input_arr=inputs, lab=labels, model=model))
                        writer.add_scalar(f'Accuracy for phase {phase} by batch number', preds.eq(labels).sum()/batch_size, batch_num)
                        writer.add_scalar(f'Average loss for phase {phase} by batch number', loss.item(), batch_num)

                    running_corrects = running_corrects + preds.eq(labels).sum()
                    running_loss = running_loss + (loss.item()*inputs.size(0))

                if phase=='train' and (mode_scheduler is not None):
                    mode_scheduler.step()

                '''Writer functions for epoch'''
                epoch_loss = running_loss / dataset_size[phase]
                print(f'Size of dataset for phase {phase}', dataset_size[phase])
                epoch_acc = running_corrects / dataset_size[phase]
                writer.add_scalar(f'Accuracy by epoch phase {phase}', epoch_acc, epoch)
                print(f'{phase.title()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
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

    model_tr = train_model(mode_scheduler=scheduler, split_in_datset=True, image_type='image')





    # train_iterator = iter(train_loader)
    # img, label = train_iterator.next()
    # img_grid = torchvision.utils.make_grid(img)
    # writer = SummaryWriter()
    # writer.add_image('test_run', img_grid)
    
    # torchbearer_trial = Trial(model=model, optimizer=optimizer, criterion=criterion, metrics=['acc'], callbacks=[TensorBoard(write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])
    # torchbearer_trial.with_generators(train_generator=train_loader, val_generator=test_loader)
    # torchbearer_trial.run(epochs=1)

    

    


    
