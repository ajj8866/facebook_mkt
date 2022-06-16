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



##########################################################################################################################################
##########################################################################################################################################

class ProductDescp:
    def __init__(self):
        prod_class = CleanData(tab_names=['Products'])
        self.major_map_encoder = prod_class.major_map_encoder
        self.major_map_decoder = prod_class.major_map_decoder
        self.product_frame = prod_class.table_dict['Products'].copy()

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
            dum_ls = [i for i in dum_ls if i.isalpha and len(i)>2]
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
        count_dict = {i[0]: i[1] for i in count_ls.most_common(num_items)}
        return count_dict

    def get_word_set(self, col='product_description'):
        if col=='product_name':
            self.clean_prod_name(col=col)
        self.clean_prod(col=col)
        full_word_set = []
        for i in self.product_frame[col]:
            full_word_set.extend(i.split())
        full_word_set = list(set(full_word_set))
        return full_word_set ,len(full_word_set)+1


    def vocab_encoder(self, col='product_description'):
        words, vocab_size = self.get_word_set(col)
        word_encoder = defaultdict(lambda: vocab_size-1)
        current_word_encoder = {key[0]: integer for integer, key in enumerate(Counter(words).most_common()[:vocab_size-1])}
        word_decoder = dict(enumerate(words))
        word_encoder.update(current_word_encoder)
        word_encoder['~'] = vocab_size-1
        word_decoder = {val: key for key, val in word_encoder.items()}
        # word_encoder = {val: key for key, val in word_decoder.items()}
        return word_encoder, word_decoder

    def unlimited_vocab(self, context_size=2):
        vocab_encoder = self.vocab_encoder()
        encoder, decoder = vocab_encoder[0], vocab_encoder[1]
        self.clean_prod(col='product_description')
        init_ls = []
        for i in range(len(self.product_frame)):
            prod_descript = self.product_frame['product_description'].iloc[i].split()
            init_ls.append([
                [
                    list(np.array([[prod_descript[i - j], prod_descript[i+j]] for j in range(1, context_size+1)]).flatten()),
                    prod_descript[i]
                ]
                for i in range(context_size, len(prod_descript)-context_size)
            ])
        self.product_frame['orignal_feature_target'] = init_ls
        self.product_frame = self.product_frame.explode('orignal_feature_target').reset_index()
        for idx, val in enumerate(self.product_frame['orignal_feature_target']):
            if type(val) is not list:
                self.product_frame.drop(idx, axis=0, inplace=True)
        self.product_frame = pd.concat([self.product_frame, pd.DataFrame(self.product_frame['orignal_feature_target'].tolist())], axis=1)
        self.product_frame.rename(columns={0: 'context', 1: 'target'}, inplace=True)
        self.product_frame = self.product_frame.explode('context').reset_index()
        self.product_frame['context_encoded'] = self.product_frame['context'].apply(lambda i: encoder[i])
        self.product_frame['target_encoded'] = self.product_frame['target'].apply(lambda i: encoder[i])
        return self.product_frame, encoder, decoder
        
    def limit_vocab(self, voc_lim= 1000):
        new_wrd_encoder = self.word_freq(num_items=voc_lim)
        new_wrd_encoder['~~'] = voc_lim
        new_wrd_decoder = {val: key for key, val in new_wrd_encoder.items()}
        DF, old_wrd_encoder, old_wrd_decoder = self.unlimited_vocab()
        df = DF.loc[:, ['context', 'target', 'context_encoded', 'target_encoded']].copy()
        context_new_ls = []
        for i in df['context']:
            if i in new_wrd_encoder.keys():
                context_new_ls.append(new_wrd_encoder[i])
            else:
                context_new_ls.append(voc_lim)
        self.product_frame['context_encoded'] = context_new_ls
        target_new_ls = []
        for i in df['target']:
            if i in new_wrd_encoder.keys():
                target_new_ls.append(new_wrd_encoder[i])
            else:
                target_new_ls.append(voc_lim)
        self.product_frame['target_encoded'] = target_new_ls
        return self.product_frame, new_wrd_encoder, new_wrd_decoder

    def neg_samples(self, func= 'limit', voc_lim=10000, num_negative=2):
        if func=='limit':
            DF, encoder, decoder = self.limit_vocab(voc_lim=voc_lim)
            df = DF.copy()
        elif func=='unlimited':
            DF, encoder, decoder = self.unlimited_vocab()
            df = DF.copy()
        df['target_value'] = 1
        df.dropna(inplace= True)
        df.drop(['minor_category', 'minor_category_encoded', 'page_id', 'location', 'price', 'create_time', 'level_0'], axis=1, inplace=True)
        for i in range(len(df)):
            for neg in df[df['target_encoded']!=df.iloc[i]['target_encoded']].sample(n=2)['context_encoded'].values:
                dum_df = df.iloc[i].copy()
                # print('Dataframe before')
                # print(dum_df[['context', 'context_encoded', 'target_value']])
                dum_df['context_encoded'] = neg
                dum_df['context'] = decoder[neg]
                dum_df['target_value'] = 0
                print(dum_df.to_frame().T)
                df = pd.concat([df, dum_df.to_frame().T], axis=0)
        self.product_frame = df.copy()
        self.product_frame.dropna(inplace=True)
        print(len(self.product_frame))
        print(self.product_frame.head())
        print(len(self.product_frame))
        # self.product_frame.loc[:, ['context_encoded', 'target_encoded', 'target_value']].astype(int,)
        return self.product_frame
            

if __name__ == '__main__':
    ex_writer = pd.ExcelWriter(path=Path(Path.cwd(), 'data_files', 'product_description_coded.xlsx'), engine='xlsxwriter')  
    prod = ProductDescp()
    prod.neg_samples()
    new_frame = prod.neg_samples().iloc[:30000]
    write = pd.ExcelWriter(Path(Path.cwd(), 'data_files', 'negative_samples.xlsx'), engine='xlsxwriter')
    with write as writer: 
        new_frame.to_excel(writer)