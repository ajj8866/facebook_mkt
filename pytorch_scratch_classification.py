from operator import mod
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
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch import nn
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transformer = transforms.Compose([ToTensor()]), X = 'image', y = 'major_category_encoded', img_dir = Path(Path.cwd(), 'images'), img_size=224, train_proportion = 0.8, is_test = False):
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
        image_df = image_class.total_clean(size=img_size, normalize=True, mode='RGB').copy()

        '''Yielding product dataset from CleanData python script'''
        product_class = CleanData(tab_names=['Products'])
        product_class.get_na_vals(df='Products')
        products_df = product_class.table_dict['Products'].copy()

        '''Merging both the previous dataset to link image with associated product category '''
        merged_df = image_df.merge(products_df, left_on='id', right_on='id')
        filtered_df = merged_df.loc[:, ['image_id', X, re.sub(re.compile('_encoded$'), '', y), y]].copy()
        filtered_df.dropna(inplace=True)
        train_end = int(len(filtered_df)*train_proportion)
        if is_test == False:
            filtered_df = filtered_df.iloc[:train_end]
        else:
            filtered_df = filtered_df.iloc[train_end:]
        self.dataset_size = len(filtered_df)
        print('Total observations in remaining dataset: ', len(filtered_df))
        self.y = torch.tensor(filtered_df[y].values)
        self.X = filtered_df[X].values


    # Not dependent on index
    def __getitem__(self, idx): 
        if self.img_inp_type == 'image':
            try:
                self.X[idx] =  Image.open(os.path.join(self.img_dir, self.X[idx]))
                if self.transformer is not None:
                    self.X[idx] = self.transformer(self.X[idx])
            except TypeError:
                self.X[idx] = self.X[idx]
        elif self.img_inp_type == 'image_array':
            try:
                print(type(self.X))
                self.X[idx] = torch.from_numpy(self.X[idx])
            except TypeError:
                self.X[idx] = self.X[idx]
        else:
            self.X[idx] = self.X[idx]        
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


class Net(nn.Module):
    def __init__(self, num_features=13): #,  pool1 = 2, pool2 =2, pool3 =2):
        super(Net, self).__init__()
        ''' Features (Convolution and Pooling) '''
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )

        '''Classification '''
        self.classifier = nn.Sequential(nn.Linear(111*111*16, 1000), #dimensions[0][0]*dimensions[0][1]/(2*2)
        nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(1000, 2000), nn.ReLU(inplace=True), 
        nn.Linear(2000, num_features))
        
    def forward(self, x):
        #print(self.dimensions)
        x = self.features(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    tot = 0
    cor = 0
    pred_list = []
    opt = torch.optim.SGD
    model = Net(num_features=14)
    train_prop = 0.8

    train_transformer = transforms.Compose([transforms.RandomRotation(40), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=False, train_proportion=train_prop)

    test_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_dataset = Dataset(transformer=train_transformer, X='image', img_size=224, is_test=True, train_proportion=train_prop)

    batch_size = 32
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    data_loader_dict = {'train': train_loader, 'eval': test_loader}
    optimizer =  opt(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_size = train_dataset.dataset_size
    test_size = test_dataset.dataset_size
    print(train_size)
    print(test_size)
    dataset_size = {'train': train_size, 'eval': test_size}

    mod_optimizer = opt(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    for epoch in range(3):
        tot = 0
        cor = 0
        model.train()
        for bch, (inp, lab) in enumerate(train_loader, start=1):
            mod_optimizer.zero_grad()
            outputs = model(inp)
            loss = criterion(outputs, lab)
            print(loss)
            loss.backward()
            mod_optimizer.step()
        

        model.eval()
        for bch, (inp_tst, lab_tst) in enumerate(test_loader, start=1):
            eval_cor = 0
            out_tst = model(inp_tst)
            pred = torch.max(out_tst.data, 1)
            pred_list.append(pred)
            tot += lab_tst.size(*0)
            cor += (pred==lab_tst).sum()
            writer.add_scalar(f'Accuracy for evaluation phase and epoch {epoch}', eval_cor/batch_size, bch)
        
        writer.add_scalar(f'Accuracy for evaluation phase epoch', eval_cor/test_size, bch)
    

    print(pred_list)
