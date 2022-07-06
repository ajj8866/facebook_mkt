import enum
from collections import Counter, defaultdict
from ntpath import join
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

def get_label_lim(df=None, cutoff_lim = 20):
    '''Gets the number of unique labels remaining in the dataset'''
    if df is None:
        merged_class = MergedData()
        merged_df = merged_class.merged_frame
    else:
        merged_df = df.copy()
    merged_df.dropna(inplace=True)
    lookup_group = merged_df.groupby(['minor_category_encoded'])['minor_category_encoded'].count()
    filt = lookup_group[lookup_group>cutoff_lim].index
    merged_df = merged_df[merged_df['minor_category_encoded'].isin(filt)]
    print(len(merged_df['minor_category_encoded'].unique()))
    return len(merged_df['minor_category_encoded'].unique())


pd.options.display.max_colwidth = 400
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 40)
plt.rc('axes', titlesize=12)
final_layer_num = 64

'''IMAGE MODEL'''
prod_dum = CleanData()
class_dict = prod_dum.major_map_encoder.keys()
classes = list(class_dict)
class_values = prod_dum.major_map_encoder.values()
class_encoder = prod_dum.major_map_encoder


ImageClassifier = models.resnet50(pretrained=True)
for i, param in enumerate(ImageClassifier.parameters(), start=1):
    param.requires_grad=False

def num_out(major=True, cutoff_lim = 30):
    if major==True:
        print(len(class_encoder))
        return len(class_encoder)
    else:
        print(get_label_lim(cutoff_lim=cutoff_lim))
        return get_label_lim(cutoff_lim=cutoff_lim) 

ImageClassifier.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(in_features=512, out_features=final_layer_num))


'''TEXT MODEL'''
class DescriptionClassifier(Module):
    def __init__(self, input_size=768):
        super(DescriptionClassifier, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 512, kernel_size=3, stride=1, padding=1), nn.Dropout(p=0.2),nn.LeakyReLU(inplace=True),  MaxPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), MaxPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace = True), nn.AvgPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=True), nn.Flatten(), nn.Linear(96, 128), nn.Tanh(),nn.Linear(128, final_layer_num))
    
    def forward(self, inp):
        x = self.main(inp)
        return x


'''COMBINED MODEL'''
'''Combined Model Classifier'''
class CombinedModel(nn.Module):
    def __init__(self, num_classes, input_size=768) -> None:
        super(CombinedModel, self).__init__()
        self.image_classifier = ImageClassifier
        self.text_classifier = DescriptionClassifier(input_size=input_size)
        self.main = nn.Sequential(nn.Linear(2*final_layer_num, num_classes))

    def forward(self, image_features, text_features):
        print(image_features.size())
        print(type(image_features))
        print(image_features.permute(0,3, 1, 2).size())
        print(type(image_features.permute(0,3, 1, 2)))
        image_features = image_features.permute(0, 3, 1, 2).float()
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

'''Combined Model Dataset'''
class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, transformer = transforms.Compose([ToTensor()]), X = 'image', y = 'major_category_encoded', cutoff_freq=20, img_dir = Path(Path.cwd(), 'images'), img_size=224, train_proportion = 0.8, is_test = False, max_length=20, min_count=2):
        '''
        X: Can be either 'image' if dataset to be instantiated using image object or 'image_array' if dataset to be instantiated using numpy array 
        y: Can be either 'major_category_encoded' or 'minor_category_encoded'
        '''
        ## Image Moel ##
        self.img_inp_type = X
        self.transformer = transformer
        self.img_dir = img_dir
        self.img_size = img_size
        merge_class = MergedData()
        merged_df = merge_class.merged_frame
        filtered_df = merged_df.loc[:, ['image_id', 'id', X, re.sub(re.compile('_encoded$'), '', y), y]].copy()
        filtered_df.dropna(inplace=True)
        ## Text Model 
        products_description = ProductDescpMannual()
        full_word_ls, _ = products_description.clean_prod()
        self.product_df = products_description.product_frame
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.max_length = max_length
        product_description_counter = Counter(full_word_ls)
        main_ls = []
        for i in self.product_df['product_description'].copy():
            temp_ls = i.split()
            temp_ls = [i for i in temp_ls if product_description_counter[i]>min_count]
            main_ls.append(' '.join(temp_ls))
        self.product_df['product_description'] = main_ls
        self.product_df = self.product_df.loc[:, ['id', 'product_description']]
        
        if y=='minor_category_encoded':
            lookup = filtered_df.groupby([y])[y].count()
            filt = lookup[lookup>cutoff_freq].index
            filtered_df = filtered_df[filtered_df[y].isin(filt)]
            new_sk_encoder = LabelEncoder()
            filtered_df[y] = new_sk_encoder.fit_transform(filtered_df['minor_category'])
        
        # print('Number of unique filtered observations in dataloader: ', len(filtered_df['minor_category_encoded'].unique()))
        # print('Unique categories remaining in dataloader: ', filtered_df['minor_category_encoded'].unique())
    
        train_end = int(len(filtered_df)*train_proportion)
        if is_test is not None:
            filtered_df = shuffle(filtered_df)
            filtered_df.reset_index(inplace=True)
            if is_test == False:
                print('Training')
                filtered_df = filtered_df.iloc[:train_end]
            elif is_test == True:
                filtered_df = filtered_df.iloc[train_end:]

        filtered_df = filtered_df.merge(self.product_df, left_on='id', right_on='id', how='left')
        self.dataset_size = len(filtered_df)
        self.main_frame = filtered_df
        print(self.main_frame.head())
        self.main_frame.to_excel(Path(Path.cwd(), 'data_files', 'letssee.xlsx'))
        self.new_category_encoder = [new_sk_encoder if y=='minor_category_encoded' else merge_class.major_map_encoder][0]
        self.new_category_decoder = [new_sk_encoder if y=='minor_category_encoded' else merge_class.major_map_decoder][0]
        self.label = self.main_frame[y]

        self.y = torch.tensor(self.main_frame[y].values)
        self.image = filtered_df[X].values
        self.product_description = self.main_frame['product_description'].to_list()

    # Not dependent on index
    def __getitem__(self, idx): 
        'Image Slicer'
        if self.img_inp_type == 'image':
            self.image[idx] =  self.transformer(Image.open(os.path.join(self.img_dir, self.image[idx])))
        else:
            try:
                self.image[idx] = self.transformer(self.image[idx])        
            except TypeError:
                try:
                    self.image[idx] = transforms.Compose([self.transformer.transforms[i] for i in range(len(self.transformer.transforms)-2)])(self.image[idx])
                except:
                    self.image[idx] = self.image[idx]

        'Text Slicer'
        descript = self.product_description[idx]
        bert_encoded = self.tokenizer.batch_encode_plus([descript], max_length=self.max_length, padding='max_length', truncation=True)
        bert_encoded = {key: torch.LongTensor(value) for key, value in bert_encoded.items()}
        with torch.no_grad():
            prod_description = self.model(**bert_encoded).last_hidden_state.swapaxes(1,2)

        self.prod_description = prod_description.squeeze(0)
        return self.image[idx], self.prod_description, self.y[idx]
    
    def __len__(self):
        return len(self.main_frame)

'''COMBINED DATALOADER'''
def get_loader(img = 'image_array', train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(40), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), y='minor_category_encoded', batch_size=35, split_in_dataset = True, train_prop = 0.8, max_length=30, min_count=4, cutoff_freq=30):
    if split_in_dataset == True:
        train_dataset = ImageTextDataset(transformer=train_transformer, X=img, img_size=224, y=y,is_test=False, train_proportion=train_prop, max_length=max_length, min_count=min_count, cutoff_freq=cutoff_freq)
        test_dataset = ImageTextDataset(transformer=test_transformer, X=img, img_size=224, y=y,is_test=True, train_proportion=train_prop, max_length=max_length, min_count=min_count, cutoff_freq=cutoff_freq)
        if y=='minor_category_encoded':
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
        image_dataset= ImageTextDataset(transformer=test_transformer, X = img, y=y, img_size=224, is_test=None, max_length=max_length, cutoff_freq=cutoff_freq)
        train_end = int(train_prop*image_dataset.dataset_size)
        train_dataset, test_dataset = random_split(image_dataset, lengths=[len(image_dataset.main_frame.iloc[:train_end]), len(image_dataset.main_frame.iloc[train_end:])])
        dataset_dict = {'train': train_dataset, 'eval': test_dataset}
        data_loader_dict = {i: DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']}
        if y == 'minor_category_encoded':
            pd.DataFrame(data= {'class': image_dataset.new_category_encoder.classes_, 'values': image_dataset.new_category_encoder.transform(image_dataset.new_category_encoder.classes_)}).to_excel(Path(Path.cwd(), 'data_files', 'outside_dataset_split_encoder.xlsx'))
        return len(image_dataset.main_frame.iloc[:train_end]), len(image_dataset.main_frame.iloc[train_end:]), data_loader_dict


'''COMBINE MODEL TRAINING'''
def train_model(combined_optimizer=optim.SGD, major=True, cutoff_lim=30, loss=nn.CrossEntropyLoss, batch_size=32, num_epochs=30,comb_scheduler=None, initial_lr=0.1, fin_lr=0.0001, step_interval=4, min_count=4, train_prop=0.8,
split_in_dataset=True, max_length=30, y='major_category_encoded', img='image_array', train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(40), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
    if major==True:
        assert y=='major_category_encoded'
    else:
        assert y=='minor_category_encoded'
    print('Number of unique categories remaining in training model: ', num_out(major=major, cutoff_lim=cutoff_lim))
    combined_model = CombinedModel(input_size=768, num_classes=num_out(major=major, cutoff_lim=cutoff_lim))
    optimizer =  combined_optimizer(combined_model.parameters(), lr=initial_lr)
    criterion = loss()
    if comb_scheduler is None:
        num_steps = num_epochs//step_interval
        gamma_mult = (fin_lr/initial_lr)**(1/num_steps)
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_interval, gamma=gamma_mult) 
    else:
        comb_scheduler()
    writer = SummaryWriter()
    best_accuracy = 0
    train_size, test_size, dataloader_dict = get_loader(img=img, y=y, split_in_dataset=split_in_dataset, train_prop=train_prop, min_count=min_count, max_length=max_length, cutoff_freq=cutoff_lim, batch_size=batch_size)
    dataset_size = {'train': train_size, 'eval': test_size}
    start = time.time()

    for epoch_num, epoch in enumerate(range(num_epochs), start=1):
        print('#'*30)
        print('Starting epoch number: ', epoch_num)
        for phase in ['train', 'eval']:
            if phase=='train':
                print('Phase right after phase iteration is : ', phase) 
                combined_model.train()
            else:
                combined_model.eval()
        
            running_loss = 0
            running_corrects = 0
            for batch_num, (images_chunk, text_chunk, labels) in enumerate(dataloader_dict[phase]): 
                print('Phase right after bathc iteration start: ', phase) 
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    print('Phase right after setting enabled grad: ', phase) 
                    outputs=combined_model(images_chunk, text_chunk)
                    preds = torch.argmax(outputs, dim=1)
                    print('Labels:\n', labels)
                    print('Predictions:\n', preds)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if batch_num%5==0:
                    print('Phase just before tensorboard writer: ', phase)
                    writer.add_scalar(f'Accuracy for phase {phase} by batch number', preds.eq(labels).sum()/batch_size, batch_num)
                    writer.add_scalar(f'Average loss for phase {phase} by batch number', loss.item(), batch_num)
            
                running_corrects = running_corrects + preds.eq(labels).sum()
                running_loss = running_loss + (loss.item()*(images_chunk.size(0)+text_chunk.size(0)))

            if phase=='train' and comb_scheduler is not None:
                comb_scheduler.step()
            else:
                step_scheduler.step()
            
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
    train_model(y='major_category_encoded', major=True, cutoff_lim=50)