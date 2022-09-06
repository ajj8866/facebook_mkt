import enum
from collections import Counter, defaultdict
from ntpath import join
from torch.utils.data import SubsetRandomSampler
from turtle import forward
import numpy as np
import pandas as pd
from pytest import param
from sqlalchemy import column
import xlsxwriter
import os
import seaborn as sns
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
nltk.download('omw-1.4')
from pathlib import Path
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from matplotlib.gridspec import GridSpec
import contractions 
from pathlib import Path
from skimage.color import rgb2gray
from skimage.filters import sobel
import logging
from skimage.io import imread, imshow
from skimage import io
import json
from skimage import img_as_float
from skimage.transform import rescale, resize
from itertools import product
from PIL import Image
from clean_tabular import CleanData, CleanImages, MergedData
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertModel
from torch.nn import Module
from torch import nn
import torch.optim as optim
import time 
import torchvision
from torchvision import models, datasets
from torchbearer import Trial
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torchbearer.callbacks import TensorBoard
from torch.nn import Module
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import torch.optim as optim
from copy import deepcopy
import copy
from text_model import TextDatasetBert, DescriptionClassifier, ProductDescpMannual
from pytorch_scratch_classification import Dataset as ImageDataSet
# from pytorch_image_transfer_classification import get_label_lim, get_loader #, res_model
from torch.nn import  MaxPool1d, AvgPool1d
from sklearn.utils import shuffle
from torchvision.transforms import ToTensor
from torchvision import transforms

#####################################################################################################################

logging.basicConfig(filename='main_logger.log', level=logging.WARNING)
model_logger= logging.getLogger(__name__)
model_logger.setLevel(logging.INFO)

model_stream_handler= logging.StreamHandler()
model_log_file= logging.FileHandler('mod_file.log')
model_log_formatter= logging.Formatter('Line Number:   %(lineno)d: Filename   %(filename)s  Message: %(message)s  Module Name:	%(module)s')

model_stream_handler.setFormatter(model_log_formatter)
model_log_file.setFormatter(model_log_formatter)

model_logger.addHandler(model_log_file)

#####################################################################################################################

def get_label_lim(df=None, cutoff_lim = 20):
    '''
    Gets the number of unique labels remaining in the dataset when using minor category as a prediction variable subject 
    to whether the category occurs at least the numbre of times specified in the cutoff_lim argument specified by the user

    Arguments
    -------------
    df: Dataframe containing the category columns .By default uses the dataframe derived instantiating the MergedData class 
        from the clean_tabular script
    cutoff_lim: The minimum number of times a category must appear in dataset. Must be integer 

    Returns: The number of unique categories remaining in the dataset
    '''
    if df is None:
        merged_class = MergedData() # Instantiated MergedData class
        merged_df = merged_class.merged_frame # Set the dataframe to be used to be equal to the merged_frame attribute from the MergedData class
    else:
        merged_df = df.copy()
    merged_df.dropna(inplace=True)
    lookup_group = merged_df.groupby(['minor_category_encoded'])['minor_category_encoded'].count() # Construct new dataframe with the index equal to the unique categories and values equal to the number of times the category appears in dataset
    filt = lookup_group[lookup_group>cutoff_lim].index # Obtains a list of categories from lookup_group, dropping those appearing an insufficient number of times
    merged_df = merged_df[merged_df['minor_category_encoded'].isin(filt)] # Uses the list derived in the line above to filter the primary dataframe
    print(len(merged_df['minor_category_encoded'].unique())) 
    return len(merged_df['minor_category_encoded'].unique())


pd.options.display.max_colwidth = 400
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 40)
plt.rc('axes', titlesize=12)
final_layer_num = 24

'''IMAGE MODEL'''
prod_dum = CleanData() # Instantiate CleanData class
class_dict = prod_dum.major_map_encoder.keys() # Yield the keys for major category encoder (category names)
classes = list(class_dict) 
class_values = prod_dum.major_map_encoder.values() # Yield the value encoded values of categories
class_encoder = prod_dum.major_map_encoder # Save encoder under variable name class encoder


# Uses resnet50's pretrained image data freezing all rows with regards to propogating weight adjustments 
ImageClassifier = models.resnet50(pretrained=True)
for i, param in enumerate(ImageClassifier.parameters(), start=1):
    param.requires_grad=False

def num_out(major=True, cutoff_lim = 30):
    '''
    Yields the number of categories to be calculated within the final layer of the  model depending on whether 
    the model aims to calculate the major or minor categories and if calculating the minor categories the minimum number 
    of time such a category must appear in the dataset

    Arguments
    -------------
    major: Boolean value. If set to true instruct model to predict major categories. Otherwise attempts to predict the minor 
        categories
    cutoff_lim: The minimum number of times a minor category must appear within the dataset in order for it to be considered
        when running model 
    
    Returns: Number of unique categories remaining within the dataset
    '''
    if major==True:
        print(len(class_encoder))
        return len(class_encoder)
    else:
        print(get_label_lim(cutoff_lim=cutoff_lim))
        return get_label_lim(cutoff_lim=cutoff_lim) 

## Adjusts the final layer of resnet50's image classifier to make it compatible with inputs and outputs used within the model 
ImageClassifier.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(in_features=512, out_features=final_layer_num))


'''TEXT MODEL'''
class DescriptionClassifier(Module):
    '''
    Model for text processing inheriting from pytorch's nn.Modul. Takes in a given input size and unlike the previous
    text classifier truncates the model so that the only layers right befoer the final layer considered so that the relevant network may be 
    concatenated with the network corresponding to images when constructing final model

    Arguments
    -------------
    input_size: Size of input layer. Must be adjusted on the basis of dimensionality of matrices implicitly used during the propogation process in pytorch
    '''
    def __init__(self, input_size=768):
        '''
        Defines the layers to subsequently be used over the course of the forward proceess
        '''
        super(DescriptionClassifier, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 512, kernel_size=3, stride=1, padding=1), nn.Dropout(p=0.2),nn.LeakyReLU(inplace=True),  MaxPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), MaxPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace = True), nn.AvgPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=True), nn.Flatten(), nn.Linear(160, 128), nn.Tanh(),nn.Linear(128, final_layer_num))
    
    def forward(self, inp):
        '''
        Forward phase of model. Takes in pytorch representation of text embedding to yield weights to be used within the neural network
        '''
        x = self.main(inp)
        return x


'''COMBINED MODEL'''
'''Combined Model Classifier'''
class CombinedModel(nn.Module):
    '''
    Combines the final layers corresponding to both the image (ImageClassifier) and text (DescriptionClassifier)

    Arguments
    -------------
    num_classes: The number of classes to be predicted in the final layer. In the context of the main training model iteration the values is the output of the "num_out" function defined earlier
    img_type: "image" if using actual images from data_files folder and "image_array" if using its numpy representation as in the dataframe
    input_size: Number of input layers to use

    Attributes
    -------------
    img_type: Image array or image file in jpeg format depending on the img_type option input by user
    image_classifier: Instantiation of image model classifier
    text_classifier: Instantiation of text model classifier
    main: Linear layers concatenatig the image and text classifier in order to avail of information provided by both 
        text and images 
    '''
    def __init__(self, num_classes, img_type,input_size=768) -> None:
        super(CombinedModel, self).__init__()
        self.img_type= img_type
        self.image_classifier = ImageClassifier
        self.text_classifier = DescriptionClassifier(input_size=input_size)
        self.main = nn.Sequential(nn.Linear(2*final_layer_num, num_classes))

    def forward(self, image_features, text_features):
        image_features_cp= torch.clone(image_features).float()
        model_logger.info('Image size within pytorch dataset: ', image_features.size())
        image_features_cp = self.image_classifier(image_features_cp)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features_cp, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

'''Combined Model Dataset'''
class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, transformer = transforms.Compose([ToTensor()]), X = 'image', y = 'major_category_encoded', cutoff_freq=20, img_dir = Path(Path.cwd(), 'images'), img_size=224, train_proportion = 0.8, is_test = False, max_length=20, min_count=2):
        '''
        Dataset inheriting from pytorch's dataset primitive for storing explanatory and target features. Given the model is desgined to combine both image
        and text data this particular dataset class instructs on how to preprocess images and text data passed in prior to it being input into the model

        Arguments
        -------------
        X: Can be either 'image' if dataset to be instantiated using image object or 'image_array' if dataset to be instantiated using numpy array 
        y: Can be either 'major_category_encoded' or 'minor_category_encoded'. 
        cutoff_freq: The minimum number of times a given category must appear within the dataset in order for it to be considered by the model 
        img_dir: The directory containing images if the "image" option passed into the X argument
        img_size: Size of image to be used within model (by default 224 to be compatible with resnet50)
        training_proportion: Proportion of data to be used as part of the trainign set (expressed as decimal )
        is_test: Boolean value indicating whether the dataset corresponds to the test data (True), training data (False) or None in the event data is not split within this class in which case the entirety of the dataset is considered
        max_length: The maximum length of each sentence to be used within the Bert model 
        min_count: THe minimum number of times a given word must appear within the product description columns in order for it be considered within the parameters of the model

        Attributes
        -------------
        img_inp_type: 'image_array' or 'image' dependnig on argumrnt passed in by user
        img_dir: Relative path from current working directory to directory containing images
        img_size: Image size
        product_df: Preprocessed product description column 
        tokenizer: Pretrained Bert tokenizer
        model: Pretrained Bert model 
        max_length: Maximum length of tokenized sentence. Sentences of a shorter length are padded while sentences of a longer length are
            truncated
        '''
        ## Image Moel ##
        self.img_inp_type = X
        self.transformer = transformer
        self.img_dir = img_dir
        self.img_size = img_size
        merge_class = MergedData()
        merged_df = merge_class.merged_frame # Uses merged_frame attribute of MergedData class to get dataframe mergingn both product and image information 
        filtered_df = merged_df.loc[:, ['image_id', 'id', X, re.sub(re.compile('_encoded$'), '', y), y]].copy() # Slices the image, id and encoded and decoded versions of either the major or minor product category columns (depending on whether y is input as "major_category_encoded" or "minor_category_encoded")
        filtered_df.dropna(inplace=True)
        ## Text Model 
        products_description = ProductDescpMannual() 
        full_word_ls, _ = products_description.clean_prod() # Uses clean_prod method of ProductDescpMannual class to preprocess
        self.product_df = products_description.product_frame # Uses the product_frame attribute of ProductDescpMannual class to yield dataframe including the preprocessed product desription columns
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Associates Bert's pretrained tokenizer to the tokenizer atttribute
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True) # Associate Bert's pretrained model with the model attribute
        self.max_length = max_length
        product_description_counter = Counter(full_word_ls) # Yields a dictionary with keys comprising of all unique words appearing in the product description column and values the number of times such words appear within the columns
        main_ls = []

        # Strips out words which appear less than the number of times specified by the user in the min_count argument
        for i in self.product_df['product_description'].copy():
            temp_ls = i.split()
            temp_ls = [i for i in temp_ls if product_description_counter[i]>min_count]
            main_ls.append(' '.join(temp_ls))
        self.product_df['product_description'] = main_ls
        self.product_df = self.product_df.loc[:, ['id', 'product_description']]
        
        # Derives a new encoder for minor categories depending if the "minor_category_encoded" option is chosen by stripping out the categories appearing less than the cutoff_freq argument set by user
        if y=='minor_category_encoded':
            lookup = filtered_df.groupby([y])[y].count()
            filt = lookup[lookup>cutoff_freq].index
            filtered_df = filtered_df[filtered_df[y].isin(filt)]
            new_sk_encoder = LabelEncoder()
            filtered_df[y] = new_sk_encoder.fit_transform(filtered_df['minor_category'])
        
        # print('Number of unique filtered observations in dataloader: ', len(filtered_df['minor_category_encoded'].unique()))
        # print('Unique categories remaining in dataloader: ', filtered_df['minor_category_encoded'].unique())
    
        train_end = int(len(filtered_df)*train_proportion)

        # Randomly samples the entire dataset splitting the values into a training and testing dataset
        if is_test is not None:
            filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
            if is_test == False:
                print('Training')
                filtered_df = filtered_df.iloc[:train_end]
            elif is_test == True:
                filtered_df = filtered_df.iloc[train_end:]

        filtered_df = filtered_df.merge(self.product_df, left_on='id', right_on='id', how='left')
        if is_test==True:
            print('#'*20)
            print('Length of test dataset: ',len(filtered_df))
            print('#'*20)
        if is_test==False:
            print('#'*20)
            print('Length of training dataset is: ', len(filtered_df))
            print('#'*20)

        self.dataset_size = len(filtered_df)
        self.main_frame = filtered_df
        # print(self.main_frame.head())
        self.main_frame.to_excel(Path(Path.cwd(), 'data_files', 'letssee.xlsx'))

        # Sets category encoder and decoder depending on which option (major or minor) user chooses
        self.new_category_encoder = [new_sk_encoder if y=='minor_category_encoded' else merge_class.major_map_encoder][0]
        self.new_category_decoder = [new_sk_encoder if y=='minor_category_encoded' else merge_class.major_map_decoder][0]
        self.label = self.main_frame[y]
        self.y = torch.tensor(self.main_frame[y].values)
        self.image = self.main_frame[X].values
        self.product_description = self.main_frame['product_description'].to_list()

    # Not dependent on index
    def __getitem__(self, idx): 
        '''
        Instructs dataset on how to index image and text observations

        Arguments
        -------------
        idx: Index position to slice

        Returns: Tubples comprising of (indexed_imaege, indexed_product_description, indexed_label) 
        '''
        'Image Slicer'
        # As with original image classifier the last two transformeations within transforms.Compose must be transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.img_inp_type == 'image':
            try:
                self.image[idx] =  self.transformer(Image.open(os.path.join(self.img_dir, self.image[idx])))
            except: 
                self.image[idx] = transforms.Compose([self.transformer.transforms[i] for i in range(len(self.transformer.transforms)-2)])(self.image[idx])
        else:
            try:
                self.image[idx] = self.transformer(self.image[idx])        
            except TypeError:
                try:
                    self.image[idx] = transforms.Compose([self.transformer.transforms[i] for i in range(len(self.transformer.transforms)-2)])(self.image[idx])
                except:
                    self.image[idx] = self.image[idx]

        'Text Slicer'
        # Uses the tokenizer set on instantiation of class to yield the tokenized version of words within product description
        descript = self.product_description[idx]
        bert_encoded = self.tokenizer.batch_encode_plus([descript], max_length=self.max_length, padding='max_length', truncation=True)
        bert_encoded = {key: torch.LongTensor(value) for key, value in bert_encoded.items()}
        with torch.no_grad():
            prod_description = self.model(**bert_encoded).last_hidden_state.swapaxes(1,2)

        self.prod_description = prod_description.squeeze(0) # Removes first dimesion to make tensor compatible with model 
        return self.image[idx], self.prod_description, self.y[idx]
    
    def __len__(self):
        '''
        Instructs datset on how to calculate length of dataset (when calling the len() function)

        Returns: Length of datsaet
        '''
        return len(self.main_frame)

'''COMBINED DATALOADER'''
def get_loader(img = 'image_array', train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(40), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), y='minor_category_encoded', batch_size=35, split_in_dataset = True, train_prop = 0.8, max_length=30, min_count=4, cutoff_freq=30):
    '''
    Uses pytorch's DataLoader to wrap an iterable around the ImageTextDataset so that the data may be passed in as batches when passing through the model trainign and evaluation loop

    Argumnets
    ------------
    img: 'image_array' or 'image' depending on whether the use the image or its numpy representation
    train_transformer: Transformations to apply to the training portion of the data
    test_transformer: Transformations to apply to the testing phase of the dataset
    split_in_dataset: Boolean: If set to True data split within the dataset and separate transformers applied to training and testing data. If set to False the test_transformer is applied
        to both training and testing phase of the data
    batch_size: Size of each batch passed into the model
    training_prop: Proportion of the data on which model will be trained
    max_length: Maximum length of sentences used in Bert tokenizer
    cutoff_freq: The minimum number of times a given minor category must appear within the dataset in order for it to be considered for the purposes of the prediction model
    min_count: The minimum number of times a given word must appear in the product description. Words occuring less than the number of times sepcifed are ommited for analytical purposes
    '''
    if split_in_dataset == True:
        train_dataset = ImageTextDataset(transformer=train_transformer, X=img, img_size=224, y=y, is_test=False, train_proportion=train_prop, max_length=max_length, min_count=min_count, cutoff_freq=cutoff_freq)
        test_dataset = ImageTextDataset(transformer=test_transformer, X=img, img_size=224, y=y, is_test=True, train_proportion=train_prop, max_length=max_length, min_count=min_count, cutoff_freq=cutoff_freq)
        dataset_dict = {'train': train_dataset, 'eval': test_dataset}
        data_loader_dict = {i: DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']}
        return train_dataset.dataset_size, test_dataset.dataset_size, data_loader_dict
    else:
        image_text_dataset= ImageTextDataset(transformer=test_transformer, X = img, y=y, img_size=224, is_test=None, max_length=max_length, cutoff_freq=cutoff_freq)
        main_df = image_text_dataset.main_frame
        main_df = main_df.sample(len(main_df))
        dataset_size = len(image_text_dataset.dataset_size)
        dataset_indices = list(range(dataset_size))
        train_end = int(train_prop*dataset_size)
        train_portion = dataset_indices[:train_end]
        test_portion = dataset_indices[train_end:]
        dataset_sizes = {'train': train_portion, 'eval': test_portion}
        data_loader_dict = {i: DataLoader(dataset=image_text_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(dataset_sizes[i])) for i in dataset_sizes}
        return len(dataset_size['train']), len(dataset_size['eval']), data_loader_dict


'''COMBINE MODEL TRAINING'''
def train_model(combined_optimizer=optim.SGD, major=True, cutoff_lim=30, loss=nn.CrossEntropyLoss, batch_size=32, num_epochs=20, comb_scheduler=None, initial_lr=0.1, fin_lr=0.0001, step_interval=2, min_count=4, train_prop=0.8,
split_in_dataset=True, max_length=30, y='major_category_encoded', img='image_array', train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(40), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
    '''
    Function used to iterate through batches as prescribed by the paramters in the get_loader for a number of epochs specified by the user. With each batch passed in the training data is used to update the weights
    corresponding to the nodes specified in the final layers of the image and text neural network, given the gradient computed on each iteration. Pytorch set to eval mode for the testing set and weights derived in prior
    training phase simply applied to datapoints within the testing to gauge how such weights perform out-of-sample

    Arguments
    -------------
    combined_optimizer: The optimizer to use (SGD by default)
    major: If set to "major" attempts to predict the major category and if set to "minor" attempts to predict the minor categories within the dataset provided such categories meet the criteria
        specified by the cutoff_lim argument 
    loss: Loss type (CrossEntropyLoss by default)
    num_epochs: Number of epochs to use i.e., number of times the model should iterate through the entirety of the dataset
    comb_scheduler: The learning rate scheduler to use. If none specidied custom scheuler which decreases exponentially at a fixed rate from the initial learning (initial_lr) to the final learning
        rate (fin_lr) is used with the learning rate adjusted after a number of periods specified by the step_interval argument
    min_count: The minimum number of times a given word must appear in the product description column in order to be used as part of the vocabulary 
    training_prop: Training proportion of data
    split_in_dataset: Whether or not to split within the Dataset class 
    max_length: Length of sentence to be used when encoding using pretrained Bert tokenizer
    img: "image" if using jpeg images in data_files directory and "image_array" if using the numpy representation of the given images
    train_transformer: Transformations to use for trainnign proportion of the dataset
    test_transformer; Transformations to use for testing proportion of the dataset

    Returns: Trained model in addition to saving model weights under the namge image_text_combined.pt in current working directory  
    '''
    if major==True:
        assert y=='major_category_encoded'
    else:
        assert y=='minor_category_encoded'
    print('Number of unique categories remaining in training model: ', num_out(major=major, cutoff_lim=cutoff_lim)) # Prints out the number of outputs to be predicted in final layuer of model 
    combined_model = CombinedModel(input_size=768, num_classes=num_out(major=major, cutoff_lim=cutoff_lim), img_type=img) # Instantiates combined model 
    optimizer =  combined_optimizer(combined_model.parameters(), lr=initial_lr) # Instantiates optimizer
    criterion = loss() # Instantiates loss criteria

    # Should no scheduler be specified the learning rate decreases exponentially from initial_lr to fin_lr every step_interval steps
    if comb_scheduler is None:
        num_steps = num_epochs//step_interval
        gamma_mult = (fin_lr/initial_lr)**(1/num_steps) # Gamma factor by which the learining rate must be multiplied in order to yield a final learning rate fin_lr in the final epoch 
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_interval, gamma=gamma_mult) # Uses step scheduler to adjust learning rate 
    else:
        comb_scheduler()

    writer = SummaryWriter() # Instantiates SummaryWriter for plotting of graphs on tensorboard backend 
    best_accuracy = 0
    train_size, test_size, dataloader_dict = get_loader(img=img, y=y, split_in_dataset=split_in_dataset, train_prop=train_prop, min_count=min_count, max_length=max_length, cutoff_freq=cutoff_lim, train_transformer=train_transformer, test_transformer=test_transformer, batch_size=batch_size)
    dataset_size = {'train': train_size, 'eval': test_size}
    start = time.time()

    for epoch_num, epoch in enumerate(range(num_epochs), start=1):
        print('#'*30)
        print('Starting epoch number: ', epoch_num)
        for phase in ['train', 'eval']:
            if phase=='train':
                combined_model.train()
            else:
                combined_model.eval()
            if comb_scheduler is not None:
                print('Current learning rate: ', comb_scheduler().get_last_lr()[0])                
            elif comb_scheduler is None:
                print('Current learning rate: ', step_scheduler.get_last_lr()[0])
            running_loss = 0
            running_corrects = 0
            for batch_num, (images_chunk, text_chunk, labels) in enumerate(dataloader_dict[phase]): 
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    model_logger.info('Input image size BEFORE FORWARD: ', images_chunk.size())
                    model_logger.info('Current phase: ', phase)
                    if img == 'image_array':
                        images_chunk= images_chunk.permute(0, 3, 1, 2)
                    # print('Input chumnk tensor: ', text_chunk)
                    outputs=combined_model(images_chunk, text_chunk)
                    model_logger.info('Ouptut size AFTER FORWARD within loop: ', outputs.size())
                    preds = torch.argmax(outputs, dim=1)
                    model_log_file.flush()
                    # print('Labels:\n', labels)
                    # print('Predictions:\n', preds)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if batch_num%10==0:
                    model_logger.debug('Phase just before tensorboard writer: ', phase)
                    writer.add_scalar(f'Accuracy for phase {phase} by batch number', preds.eq(labels).sum()/batch_size, batch_num)
                    writer.add_scalar(f'Average loss for phase {phase} by batch number', loss.item(), batch_num)
            
                running_corrects = running_corrects + preds.eq(labels).sum()
                running_loss = running_loss + (loss.item()*(images_chunk.size(0)))

            if (phase=='train') and (comb_scheduler is None):
                step_scheduler.step()
            elif (phase=='train') and (comb_scheduler is not None):
                comb_scheduler.step()
            
            epoch_loss = running_loss/dataset_size[phase]
            print(f'Size of dataset for phase {phase}', dataset_size[phase])
            epoch_accuracy = running_corrects/dataset_size[phase]
            writer.add_scalar(f'Accuracy by epoch phase {phase}', epoch_accuracy, epoch)
            print(f'{phase.title()} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')
            writer.add_scalar(f'Average loss by epoch phase {phase}', epoch_loss, epoch)

            if phase=='eval' and epoch_accuracy > best_accuracy:
                best_accuracy=epoch_accuracy
                best_model_weights = copy.deepcopy(combined_model.state_dict())
                print(f'Best Accuracy Value in {epoch_num}th Epoch and equal to : {best_accuracy:.4f}')

    combined_model.load_state_dict(best_model_weights)
    torch.save(combined_model.state_dict(), 'image_text_combined.pt')
    time_diff = time.time()-start
    print(f'Time taken for model to run: {(time_diff//60)} minutes and {(time_diff%60):.0f} seconds')
    return combined_model

if __name__ == '__main__':
    train_model(y='major_category_encoded', major=True, cutoff_lim=50, img='image', split_in_dataset=True, num_epochs=50, step_interval=4, max_length=40)
