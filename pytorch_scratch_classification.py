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
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transformer = transforms.Compose([ToTensor()]), X = 'image', y = 'major_category_encoded', img_dir = Path(Path.cwd(), 'images'), img_size=128, train_end = 10000, is_test = False):
        '''
        X: Can be either 'image' if dataset to be instantiated using image object or 'image_array' if dataset to be instantiated using numpy array 
        y: Can be either 'major_category_encoded' or 'minor_category_encoded'
        '''
        self.img_inp_type = X
        self.transformer = transformer
        self.img_dir = img_dir
        self.img_size = img_size
        '''Yielding images dataset from CleanImages python script'''
        image_class = CleanImages()
        image_df = image_class.total_clean(size=img_size, normalize=True).copy()

        '''Yielding product dataset from CleanData python script'''
        product_class = CleanData(tab_names=['products', 'products_2'])
        product_class.try_merge(['products', 'products_2'])
        product_class.get_na_vals(df='combined')
        products_df = product_class.expand_category().copy()

        '''Merging both the previous dataset to link image with associated product category '''
        merged_df = image_df.merge(products_df, left_on='id', right_on='id')
        filtered_df = merged_df.loc[:, ['image_id', X, re.sub(re.compile('_encoded$'), '', y), y]].copy()
        filtered_df.dropna(inplace=True)
        if is_test == False:
            filtered_df = filtered_df.iloc[:train_end]
        else:
            filtered_df = filtered_df.iloc[train_end:]
        print('Total observations in remaining dataset: ', len(filtered_df))
        self.y = torch.tensor(filtered_df[y].values)
        self.X = filtered_df[X].values


    # Not dependent on index
    def __getitem__(self, idx): 
        if self.img_inp_type == 'image':
            self.X[idx] =  Image.open(os.path.join(self.img_dir, self.X[idx]))
            print(self.X[idx])
            if self.transformer is not None:
                self.X[idx] = self.transformer(self.X[idx])
        elif self.img_inp_type == 'image_array':
            print(self.X)
            print(type(self.X))
            self.X[idx] = torch.from_numpy(self.X[idx])
        else:
            self.X[idx] = self.X[idx]        
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


class Net(nn.Module):
    def __init__(self, num_features=13): #,  pool1 = 2, pool2 =2, pool3 =2):
        super(Net, self).__init__()
        ''' Features (Convolution and Pooling) '''
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )

        '''Classification '''
        self.classifier = nn.Sequential(nn.Linear(32*32*64, 1000), #dimensions[0][0]*dimensions[0][1]/(2*2)
        nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(1000, 2000), nn.ReLU(inplace=True), 
        nn.Linear(2000, num_features))
        
    def forward(self, x):
        #print(self.dimensions)
        x = self.features(x)
        x = x.reshape(-1, 16*63*63)
        x = self.classifier(x)
        return x

        
    