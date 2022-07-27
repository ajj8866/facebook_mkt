from matplotlib.transforms import Transform
from clean_tabular import CleanData, CleanImages, MergedData
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import multiprocessing
from torchvision.transforms import ToTensor
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from sklearn.preprocessing import LabelEncoder
from torch import nn
import re
import numpy as np
import os
from pathlib import Path
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transformer = transforms.Compose([ToTensor()]), X = 'image', y = 'major_category_encoded', cutoff_freq=20, img_dir = Path(Path.cwd(), 'images'), img_size=224, train_proportion = 0.8, is_test = False):
        '''
        X: Can be either 'image' if dataset to be instantiated using image object or 'image_array' if dataset to be instantiated using numpy array 
        y: Can be either 'major_category_encoded' or 'minor_category_encoded'
        '''
        self.img_inp_type = X
        self.transformer = transformer
        self.img_dir = img_dir
        self.img_size = img_size
        merge_class = MergedData()
        merged_df = merge_class.merged_frame
        filtered_df = merged_df.loc[:, ['image_id', X, re.sub(re.compile('_encoded$'), '', y), y]].copy()
        filtered_df.dropna(inplace=True)
        print(filtered_df[y].value_counts())
        print(filtered_df[re.sub(re.compile('_encoded$'), '', y)].value_counts())
        if y=='minor_category_encoded':
            lookup = filtered_df.groupby([y])[y].count()
            filt = lookup[lookup>cutoff_freq].index
            filtered_df = filtered_df[filtered_df[y].isin(filt)]
            new_sk_encoder = LabelEncoder()
            filtered_df[y] = new_sk_encoder.fit_transform(filtered_df['minor_category'])
        print('Number of Unique Categories Remaining: ', len(filtered_df[y].unique()))
        train_end = int(len(filtered_df)*train_proportion)
        if is_test is not None:
            filtered_df = shuffle(filtered_df)
            filtered_df.reset_index(inplace=True)
            print('NEW FILTERED DF\n', filtered_df.head())
            if is_test == False:
                print('Training')
                filtered_df = filtered_df.iloc[:train_end]
            elif is_test == True:
                filtered_df = filtered_df.iloc[train_end:]
        self.dataset_size = len(filtered_df)
        self.all_data = filtered_df
        print('filtered_df: ', print(type(self.all_data)))
        print(self.all_data)
        print('Total observations in remaining dataset: ', len(filtered_df))
        self.new_category_encoder = new_sk_encoder
        self.y = torch.tensor(filtered_df[y].values)
        self.X = filtered_df[X].values

    # Not dependent on index
    def __getitem__(self, idx): 
        if self.img_inp_type == 'image':
            self.X[idx] =  self.transformer(Image.open(os.path.join(self.img_dir, self.X[idx])))
        else:
            try:
                self.X[idx] = self.transformer(self.X[idx])        
            except TypeError:
                print(self.X[idx])
                print(type(self.X[idx]))
                try:
                    self.X[idx] = transforms.Compose([self.transformer.transforms[i] for i in range(len(self.transformer.transforms)-2)])(self.X[idx])
                except:
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
 #    StepLR(optimizer=optimizer, step_size=1, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    train_size = train_dataset.dataset_size
    test_size = test_dataset.dataset_size
    print(train_size)
    print(test_size)
    dataset_size = {'train': train_size, 'eval': test_size}

    mod_optimizer = opt(model.parameters(), lr=0.001)
    writer = SummaryWriter()

    for epoch in range(15):
        running_train_correct = 0
        running_train_loss = 0
        model.train()
        for bch, (inp, lab) in enumerate(train_loader, start=1):
            mod_optimizer.zero_grad()
            outputs = model(inp)
            batch_loss = criterion(outputs, lab)
            train_corrects = torch.argmax(outputs, dim=1).eq(lab).sum()
            writer.add_scalar('Training accuracy for batch', train_corrects/batch_size, bch)
            writer.add_scalar('Training loss for batch', batch_loss.item()/batch_size, bch)
            running_train_loss = running_train_loss + batch_loss.item()
            running_train_correct = running_train_correct + train_corrects
            batch_loss.backward()
            mod_optimizer.step()
        
        writer.add_scalar('Training accuracy by epoch', running_train_correct/train_size, epoch)
        writer.add_scalar('Average training loss by epocy', running_train_loss/train_size, epoch)
        

        model.eval()
        running_test_correct = 0
        running_test_loss = 0
        label_ls = []
        for bch, (inp_tst, lab_tst) in enumerate(test_loader, start=1):
            out_tst = model(inp_tst)
            pred = torch.argmax(out_tst, dim=1)
            test_loss = criterion(out_tst, lab_tst)
            eval_corr = pred.eq(lab_tst).sum()
            running_test_correct = running_test_correct + eval_corr
            pred_list.append(pred)
            label_ls.append(lab_tst)
            running_test_loss = running_test_loss + test_loss.item()
            writer.add_scalar('Average evaluation loss by batch number', test_loss.item()/batch_size, bch)
            writer.add_scalar('Evaluation accuracy by batch number', eval_corr/batch_size, bch)
        
        writer.add_scalar('Average loss for evaluation phase by epoch ', running_test_loss/test_size, epoch)
        writer.add_scalar(f'Accuracy for evaluation phase epoch', eval_corr/test_size, epoch)
    

    print(pred_list)
