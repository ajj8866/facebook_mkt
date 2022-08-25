from pathlib import Path
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
from sklearn.preprocessing import LabelEncoder
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

    res_model = models.resnet50(pretrained=True) # Defines pretrained resnet50 model as model of choice

    # Freezes gradients for all layers so that gradients are not adjusted during training phase
    for i, param in enumerate(res_model.parameters(), start=1):
        param.requires_grad=False


    def get_label_lim(cutoff_lim = 20):
        '''
        Function to be used when categorising minor categories. Sets the minimum number of times a given minor cateogry must appear in dataset
        SHould the category appear less than the number of times stipulated in the cutoff_lim argument the the obseration is dropped
        '''
        merged_class = MergedData() # Instantiates MergedData class to yield a dataframe merging the product and images dataframe 
        merged_df = merged_class.merged_frame # Assigns the dataframe from the merged_frame instance attribute of the MergedData class to the merged_df variable 
        merged_df.dropna(inplace=True)

        # Constructs dataframe using the minor product category as the index and the respective count of the product as the value
        lookup_group = merged_df.groupby(['minor_category_encoded'])['minor_category_encoded'].count() 

        filt = lookup_group[lookup_group>cutoff_lim].index # Yields the index values (product categories) appearnign more than the amount of times specified by the cutoff_lim argument
        merged_df = merged_df[merged_df['minor_category_encoded'].isin(filt)] # Filters the dataframe so that only those product categories meeting criteria are retained 
        print(len(merged_df['minor_category_encoded'].unique())) # Displays the number of unique minor product categories remaining to be used as the output layer of subsequent model 
        return len(merged_df['minor_category_encoded'].unique()) # Returns the number of categories (potential target variables for purposes of the model)


    # Adjust the final (fc) layer of resnet50's model so that is compatible for the specific use case of the model in additon to addinng in certain non-linearities
    # in order to make the model more robust
    res_model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(in_features=512, out_features=get_label_lim())) #, nn.Linear(in_features=64, out_features=13))

    # Set the default optimiser to be passed into model with an initial learning rate of 0.1
    opt = optim.SGD 
    optimizer =  opt(res_model.parameters(), lr=0.1) 

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5, 10, 15, 20, 25, 30], gamma=0.3) # Set a learning rate scheduler 
    criterion = nn.CrossEntropyLoss() # Default loss metric to be passed into the model 

    def get_loader(img = 'image_array', y='minor_category_encoded', batch_size=35, split_in_dataset = True, train_prop = 0.8):
        '''
        Function used to process the dataset into batches using pytorch's DataLoader dividing it into training and testing set 

        img: Must be one of "image" should the user wish to upload images directly from the data_files directory or "image_array" should 
            user wish to use the images numpy representation as in the dataframe 
        y: Must be one of "minor_category_encoded" if model is being used to classify the minor categories or "major_category_encoded"
           should model be used to classify the majo categories
        batch_size: Batch size to be used
        split_in_dataset: Boolean value: If set to True data randomly split into training and testing within the Dataset class from the
            pytorch_scratch_classification script with additional transformations applied to the training dataset and if set to flase split remains
            random but a basic transformation is applied to both the training and testing dataset though the model runs faster
        training_prop: The proportion of the dataset to be used as training

        Returns:
        1) Length of training set
        2) Legnth of testing set
        3) Dictionary with keys equal to "train" (training data), "evel" (testing data) with values equal to the dataloader corresponding to training data
            testing data, respectively 
 
        '''

        # For both of the followig tranformations transforms.ToTensor() and transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225] are required 
        # in order to render the values passed into the model compatible with resnet50
        train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(40), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # Images randomly rotated and flipped in order to synthesise a greater variety of observations on which the model can train 
        test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

        if split_in_dataset == True:
            train_dataset = Dataset(transformer=train_transformer, X=img, img_size=224, y=y,is_test=False, train_proportion=train_prop) # Uses train_transformer and is_test set to False within the Dataset class 
            test_dataset = Dataset(transformer=test_transformer, X=img, img_size=224, y=y,is_test=True, train_proportion=train_prop) # Uses test_transformer and is_test set to True within the Dataset class 
            print(train_dataset.new_category_encoder)
            print(dict(zip(train_dataset.new_category_encoder.classes_, train_dataset.new_category_encoder.transform(train_dataset.new_category_encoder.classes_))))
            print(dict(zip(test_dataset.new_category_encoder.classes_, test_dataset.new_category_encoder.transform(test_dataset.new_category_encoder.classes_))))
            pd.DataFrame(data={'classes': test_dataset.new_category_encoder.classes_, 'values': test_dataset.new_category_encoder.transform(test_dataset.new_category_encoder.classes_)}).to_excel(Path(Path.cwd(), 'data_files', 'test_transformer.xlsx'))
            pd.DataFrame(data={'classes': train_dataset.new_category_encoder.classes_, 'values': train_dataset.new_category_encoder.transform(train_dataset.new_category_encoder.classes_)}).to_excel(Path(Path.cwd(), 'data_files', 'train_transformer.xlsx'))
            #assert( (pd.DataFrame(data={'classes': test_dataset.new_category_encoder.classes_, 'values': test_dataset.new_category_encoder.transform(test_dataset.new_category_encoder.classes_)}) == pd.DataFrame(data={'classes': train_dataset.new_category_encoder.classes_, 'values': train_dataset.new_category_encoder.transform(train_dataset.new_category_encoder.classes_)})).all() )
            dataset_dict = {'train': train_dataset, 'eval': test_dataset}
            data_loader_dict = {i: DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']}
            return train_dataset.dataset_size, test_dataset.dataset_size, data_loader_dict
        else:
            image_dataset= Dataset(transformer=test_transformer, X = img, y=y, img_size=224, is_test=None)
            train_end = int(train_prop*image_dataset.dataset_size) # Ending index of training set
            train_dataset, test_dataset = random_split(image_dataset, lengths=[len(image_dataset.all_data.iloc[:train_end]), len(image_dataset.all_data.iloc[train_end:])]) # Pytorch's random_split applied to segregate the dataset into training and testing
            dataset_dict = {'train': train_dataset, 'eval': test_dataset}
            data_loader_dict = {i: DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']} # Dictionary using dataloader correspoinding to training and testing as values
            pd.DataFrame(data= {'class': image_dataset.new_category_encoder.classes_, 'values': image_dataset.new_category_encoder.transform(image_dataset.new_category_encoder.classes_)}).to_excel(Path(Path.cwd(), 'data_files', 'outside_dataset_split_encoder.xlsx'))
            return len(image_dataset.all_data.iloc[:train_end]), len(image_dataset.all_data.iloc[train_end:]), data_loader_dict
        
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
        '''
        Displays images taking for corresponding torch tensors 
        '''
        input_ten = torch.clone(input_ten_orig)

        # Undo initial transformation which was used to convert images into a form compatible with resnet50
        inv_normalize_array = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255]) 
        inv_normalize = transforms.Compose([inv_normalize_array])
        input_ten = inv_normalize(input_ten)

        input_numpy = input_ten.numpy() # Convert torchh tensor into numpy array 

        # Torch tensor represnetation of images is in the form (channel)x(height)x(width) while numpy representation of image is in the form 
        # (height)x(width)x(channel) so the following transpose must be applied so that plt correctly recognises the numpy array as an image 
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
        '''
        Displys the first three observations from the batch along with their actual prodcuct category, along with the category as predicted
        by the model and associated probability as per the model 

        input_arr: Numpy array to be passed in 
        lab: 
        mod: Model to be used when making predictions
        '''
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
        '''
        model: Base model to use during the iteration process (by default equal to `res_model`
        optimizer: Optimiser to use (by default set to `opt`)
        loss_type:  Loss metric to use (by default set to `criterion`)
        num_epochs: Number of epochs to use
        model_scheduler: Learning rate scheduler to use
        batch_size: Batch size to pass in
        image_type: Must be one of `"image_array"` or `"image"` depending on model should use actual images or their numpy representation (processed using the `Dataset` class)
        split_in_datset: Boolean variable. If set to True the dataset is split within the `Dataset` class and a series of random flips and rotations applied to the training dataset. 
            If set to False the dataset is split within the `get_loader`function and identical transformations applied to both the training and testing set but model runs faster
        '''
        best_model_weights = copy.deepcopy(model.state_dict()) #May be changed at end of each "for phase block"
        best_accuracy = 0 # May be changed at end of each "for phase block"
        start = time.time()
        writer = SummaryWriter() # Instantitates tensorboard backend 

        # Yields the lenght of the training set, testing set and dictioanry with key value pairs corresponding to a train  and testing label
        # and their corresponding dataloader objects as values
        train_size, test_size, data_loader_dict = get_loader(batch_size=batch_size, img=image_type, split_in_dataset=split_in_datset)

        dataset_size = {'train': train_size, 'eval': test_size} # Dictionary with of respective training and testing lengths as values (for convenience when using in subsequent model loop)
    
        for epoch in range(num_epochs):
            for phase in ['train', 'eval']:
                if phase == 'train':
                    model.train() # If in training phase instructs the model to adjust weights and continue using transformations the likes of dropout, flips and rotations when procesing images
                else:
                    model.eval() # IF in testing pahse instructs the model not to adjust weights and refrain from using transformations the likes of dropout, flips and rotations
                
                running_loss = 0
                running_corrects = 0

                for batch_num, (inputs, labels) in enumerate(data_loader_dict[phase], start=1):
                    optimizer.zero_grad() # Gradients reset to zero at beginning of both training and evaluation phase

                    with torch.set_grad_enabled(phase == 'train'):
                        #print('Inputs: \n', inputs)
                        # print(inputs.size())
                        outputs = model(inputs) # Get torch tensor output of model of shape (batch_size)x()
                        #outputs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1) # For each row within tensor select the column index containig the maximum value withini the row
                        # print('Predictions: \n', preds)
                        # print('Labels:\n', labels)
                        loss = loss_type(outputs, labels) # Calculates the loss for current batch 
                        if phase == 'train':
                            loss.backward() #Calculates gradients
                            optimizer.step() # Adjusts weight given gradient calculated with magnitude of adjustment dependinng on learning rate

                    if batch_num%20==0:
                        '''Writer functions for batch'''
                        # writer.add_figure('Predictions vs Actual',plot_classes_preds(input_arr=inputs, lab=labels, model=model))
                        writer.add_scalar(f'Accuracy for phase {phase} by batch number', preds.eq(labels).sum()/batch_size, batch_num) # Depicts accuray of current bathc on tensorbaord
                        writer.add_scalar(f'Average loss for phase {phase} by batch number', loss.item(), batch_num) # Depicts loss of current batch on tensorboard

                    running_corrects = running_corrects + preds.eq(labels).sum() # Yields the total number of correct predictions for curent epoch
                    running_loss = running_loss + (loss.item()*inputs.size(0)) # Yields the total loss for current epoch

                # Adjusts learning rate based on learning rate scheduler if one passed in as argument 
                if phase=='train' and (mode_scheduler is not None):
                    mode_scheduler.step()

                '''Writer functions for epoch'''
                epoch_loss = running_loss / dataset_size[phase] # Average loss for epoch 
                print(f'Size of dataset for phase {phase}', dataset_size[phase])
                epoch_acc = running_corrects / dataset_size[phase] # Proportion of correct preditions made within current epoch
                writer.add_scalar(f'Accuracy by epoch phase {phase}', epoch_acc, epoch) # Depicts current epoch accuracy on tensorboard
                print(f'{phase.title()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                writer.add_scalar(f'Average loss by epoch phase {phase}', epoch_loss, epoch) # Depicts current epoch's loss

                # Should the accuracy in currnet epoch be greater than that of all previous epoch the weights corresponding to the epoch
                # are saved in the best_model_weights variables
                if phase == 'eval' and epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc 
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print(f'Best val Acc: {best_accuracy:.4f}')


        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), 'image_model.pt') # Weights corresponding to the best performing epoch are saved as file named 'image_model.pt'
        time_diff = time.time()-start
        print(f'Time taken for model to run: {(time_diff//60)} minutes and {(time_diff%60):.0f} seconds')
        return model

    model_tr = train_model(mode_scheduler=scheduler, split_in_datset=False, image_type='image_array')





    # train_iterator = iter(train_loader)
    # img, label = train_iterator.next()
    # img_grid = torchvision.utils.make_grid(img)
    # writer = SummaryWriter()
    # writer.add_image('test_run', img_grid)
    
    # torchbearer_trial = Trial(model=model, optimizer=optimizer, criterion=criterion, metrics=['acc'], callbacks=[TensorBoard(write_graph=False, write_batch_metrics=True, write_epoch_metrics=False)])
    # torchbearer_trial.with_generators(train_generator=train_loader, val_generator=test_loader)
    # torchbearer_trial.run(epochs=1)

    

    


    
