import enum
from collections import Counter, defaultdict
from ntpath import join
import numpy as np
import pandas as pd
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
<<<<<<< HEAD
import pickle
=======
>>>>>>> aec0a94e116a4de964e5f4f1cef8881643d60fde
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
from clean_tabular import CleanData, CleanImages
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertModel
from torch.nn import Module
from torch import nn
import torch.optim as optim
import time 
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
from copy import deepcopy
import copy

pd.set_option('max_colwidth', 200)
pd.set_option('display.max_columns', 20)
class ProductDescpMannual(CleanData):
    def __init__(self):
        super().__init__(tab_names=['Products'])
        self.label_len = len(self.major_map_decoder)
        self.product_frame = self.table_dict['Products'].copy()

    def clean_prod(self, col='product_description'):
        stop_words = set(stopwords.words('english'))
        word_lemmitizer = WordNetLemmatizer()
        punct_re = re.compile(r'[^A-Za-z \n-]') #[^A-Za-z0-9 .]
        self.product_frame[col] = self.product_frame[col].copy().str.replace(punct_re, '')
        all_text_ls = []
        main_ls = []
        for i in self.product_frame[col]:
            dum_ls = i.split()
            dum_ls = [i for i in dum_ls if i not in  stop_words]
            dum_ls = [i.replace('\n', ' ') for i in dum_ls]
            dum_ls = [i.strip('\-') for i in dum_ls]
            dum_ls = [word_lemmitizer.lemmatize(i) for i in dum_ls]
            dum_ls = [contractions.fix(j) for j in dum_ls]
            dum_ls = [i for i in dum_ls if i.isalpha] # and len(i)>2
            main_ls.append(' '.join(dum_ls))
            # print(' '.join(dum_ls))
            all_text_ls.extend(dum_ls)
        all_text_ls = [i.lower() for i in all_text_ls]
        self.product_frame[col] = main_ls 
        self.product_frame[col] = self.product_frame[col].apply(lambda i: i.lower())
        print(len(all_text_ls))
        return all_text_ls, len(all_text_ls)
    
    def word_freq(self, col='product_description', num_items=100):
        all_words = self.clean_prod(col=col)[0]
        count_ls = Counter(all_words)
        if num_items is not None:
            count_dict = {i[0]: i[1] for i in count_ls.most_common(num_items)}
        else:
            count_dict = {i[0]: i[1] for i in count_ls.most_common()}
        return count_dict

    def get_word_set(self, col='product_description'):
        if col=='product_name':
            self.clean_prod_name(col=col)
        self.clean_prod(col=col)
        self.full_word_ls = []
        for i in self.product_frame[col]:
            self.full_word_ls.extend(i.split())
        self.full_word_set = list(set(self.full_word_ls))
        return self.full_word_ls, self.full_word_set ,len(self.full_word_set)+1

    def vocab_encoder(self, col='product_description', limit_it=False, vocab_limit=30000):
        word_ls, word_set, full_vocab_size = self.get_word_set(col)
        if limit_it==False:
            vocab_size = full_vocab_size
            self.word_encoder = defaultdict(lambda: vocab_size-1)
            current_word_encoder = {key[0]: integer for integer, key in enumerate(Counter(word_ls).most_common()[:vocab_size-1])}
            self.word_encoder['<UNKNOWN>'] = vocab_size-1
            self.word_encoder.update(current_word_encoder)
            self.word_decoder = {val: key for key, val in self.word_encoder.items()}
        else:
            vocab_size=vocab_limit
            self.word_encoder = defaultdict(lambda: vocab_size)
            current_word_encoder = {key[0]: integer for integer, key in enumerate(Counter(word_ls).most_common()[:vocab_size])}
            self.word_encoder['<UNKNOWN>'] = vocab_size
            self.word_encoder.update(current_word_encoder)
            self.word_decoder = {val: key for key, val in self.word_encoder.items()}
        
        # self.word_encoder = {val: key for key, val in self.word_decoder.items()}
        return self.word_encoder, self.word_decoder, self.product_frame

    
    def dataloader_preprocess(self, limit_it=True, vocab_lim=10000, context_size=5):
        product_encoder, product_decoder, DF = self.vocab_encoder(limit_it=limit_it, vocab_limit=vocab_lim)
        df = DF.copy()
        init_ls = []
        for i in range(len(df)):
            prod_descript = df['product_description'].iloc[i].split()
            init_ls.append([
                [
                    list(np.array([[prod_descript[i - j], prod_descript[i+j]] for j in range(1, context_size+1)]).flatten()),
                    prod_descript[i]
                ]
                for i in range(context_size, len(prod_descript)-context_size)
            ])
        df['original_context_target'] = init_ls
        df = df.explode('original_context_target').reset_index()
        for idx, val in enumerate(df['original_context_target']):
            if type(val) is not list:
                df.drop(idx, axis=0, inplace=True)
        
        def stub_coder(ls, code_type='encoded'):
            code_dict = {'encoded': product_encoder, 'decoded': product_decoder}
            new_ls = []
            for i in ls.copy():
                new_ls.append(code_dict[code_type][i])
            return new_ls

        df = pd.concat([df, pd.DataFrame(df['original_context_target'].tolist())], axis=1)
        df.rename(columns={0: 'context', 1: 'target'}, inplace=True)
        df.dropna(inplace=True)
        df['context_encoded'] = df['context'].apply(lambda i: stub_coder(ls=i, code_type='encoded'))
        df['context_decoded'] = df['context_encoded'].apply(lambda i: stub_coder(ls=i,code_type='decoded'))
        df['target_encoded'] = df['target'].apply(lambda i: product_encoder[i])
        df['target_decoded'] = df['target_encoded'].apply(lambda i: product_decoder[i])
        # print(df)
        # print(df.columns)
        self.product_frame = df.copy()
        return self.product_frame, product_encoder, product_decoder

class TextDatasetBert(torch.utils.data.Dataset):
    def __init__(self, max_length=50, min_count=2):
        prod = ProductDescpMannual()
        full_word_ls, _ = prod.clean_prod()
        print(prod.product_frame['product_description'].head())
    
        # current_vocab = prod.full_word_set
        # currnet_corpus = prod.full_word_ls
        # current_vocab_size = len(current_vocab) + 1
        # self.product_df, self.mannual_word_encoder, self.mannual_word_decoder = prod.dataloader_preprocess()
        self.product_df = prod.product_frame


        self.label_encoder = prod.major_map_encoder
        self.label_decoder = prod.major_map_decoder
        self.labels = self.product_df['major_category_encoded'].to_list()

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
        self.product_descriptions = self.product_df['product_description'].to_list()
        

    def __getitem__(self, idx):
        label = torch.as_tensor(self.labels[idx])
        descript = self.product_descriptions[idx]
        bert_encoded = self.tokenizer.batch_encode_plus([descript], max_length=self.max_length, padding='max_length', truncation=True)
        bert_encoded = {key: torch.LongTensor(value) for key, value in bert_encoded.items()}
        with torch.no_grad():
            new_description = self.model(**bert_encoded).last_hidden_state.swapaxes(1,2)
<<<<<<< HEAD

=======
        
>>>>>>> aec0a94e116a4de964e5f4f1cef8881643d60fde
        new_description = new_description.squeeze(0)
        return new_description, label

    def __len__(self):
        return len(self.labels)


from torch.nn import  MaxPool1d, AvgPool1d

class DescriptionClassifier(Module):
    def __init__(self, input_size=768, num_classes=13):
        super(DescriptionClassifier, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 512, kernel_size=3, stride=1, padding=1), nn.Dropout(p=0.2),nn.LeakyReLU(inplace=True),  MaxPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), MaxPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace = True), nn.AvgPool1d(kernel_size=2, stride=2), 
        nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=True), nn.Flatten(), nn.Linear(192, 64), nn.Tanh(),nn.Linear(64, num_classes))
    
    def forward(self, inp):
        x = self.main(inp)
<<<<<<< HEAD
        print('After forward output',x)
        print('After forward output size: ', x.size())
=======
>>>>>>> aec0a94e116a4de964e5f4f1cef8881643d60fde
        return x


'Model training and testing function'
def train_model(model, optimizer, dataset,loss_type, num_epochs = 1, mode_scheduler=None, batch_size = 24, train_proportion=0.75):
    best_accuracy = 0 # May be changed at end of each "for phase block"
    start = time.time()
    writer = SummaryWriter()
    train_end = int(train_proportion*len(dataset))
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_end, len(dataset)-train_end])
    dataset_dict = {'train': train_dataset, 'eval': test_dataset}
    dataloader_dict = {i: torch.utils.data.DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']}


    for epoch in range(num_epochs):
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0
            running_corrects = 0

            for batch_num, (inputs, labels) in enumerate(dataloader_dict[phase], start=1):
                optimizer.zero_grad() # Gradients reset to zero at beginning of both training and evaluation phase

                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs)
                    # print(inputs.size())
                    outputs = model(inputs)
                    #outputs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    # print(preds)
                    loss = loss_type(outputs, labels)
                    if phase == 'train':
                        loss.backward() #Calculates gradients
                        optimizer.step()

                if batch_num%20==0:
                    '''Writer functions for batch'''
                    # writer.add_figure('Predictions vs Actual',plot_classes_preds(input_arr=inputs, lab=labels, model=model))
                    writer.add_scalar(f'Accuracy for phase {phase} by batch number', preds.eq(labels).sum()/batch_size, batch_num)
                    writer.add_scalar(f'Average loss for phase {phase} by batch number', loss.item(), batch_num)

                running_corrects = running_corrects + preds.eq(labels).sum()
                running_loss = running_loss + (loss.item()*inputs.size(0))

            if phase=='train' and (mode_scheduler is not None):
                mode_scheduler.step()

            '''Writer functions for epoch'''
            epoch_loss = running_loss / len(dataset_dict[phase])
            print(f'Size of dataset for phase {phase}', dataset_dict[phase])
            epoch_acc = running_corrects / len(dataset_dict[phase])
            writer.add_scalar(f'Accuracy by epoch phase {phase}', epoch_acc, epoch)
            print(f'{phase.title()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            writer.add_scalar(f'Average loss by epoch phase {phase}', epoch_loss, epoch)
            print(f'Done {epoch} epoch(s)')

            if phase == 'eval' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                print(f'Best val Acc: {best_accuracy:.4f}')
        


    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), 'prodcut_model.pt')
    time_diff = time.time()-start
    print(f'Time taken for model to run: {(time_diff//60)} minutes and {(time_diff%60):.0f} seconds')
    return model

<<<<<<< HEAD
print(__name__)

if __name__ == '__main__':
=======

if __name__ == '__man__':
>>>>>>> aec0a94e116a4de964e5f4f1cef8881643d60fde
    prod = ProductDescpMannual()
    prod.dataloader_preprocess()
    dataset = TextDatasetBert()
    batch_size=32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    product_model = DescriptionClassifier()
    n_epochs=10
    batch_size=32
    train_prop = 0.75
    train_end = int(train_prop*len(dataset))
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_end, len(dataset)-train_end])
    dataset_dict = {'train': train_dataset, 'eval': test_dataset}
    dataloader_dict = {i: torch.utils.data.DataLoader(dataset_dict[i], batch_size=batch_size, shuffle=True) for i in ['train', 'eval']}
    opt = optim.SGD
    optimizer =  opt(product_model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    num_epochs=30
    step_sz = 4
    num_steps = num_epochs//step_sz
    initial_lr = 0.1
    fin_lr = 0.0001
    gamma_mult = (fin_lr/initial_lr)**(1/num_steps)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=4, gamma=gamma_mult) 
    train_model(model=product_model, optimizer=optimizer, dataset=dataset, loss_type=criterion, num_epochs=num_epochs, mode_scheduler=step_scheduler)




