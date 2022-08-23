import fastapi
from fastapi import Request, File, UploadFile, Form, FastAPI
from fastapi.responses import JSONResponse
import requests
from pathlib import Path

import uvicorn
from clean_tabular import CleanData, CleanImages, MergedData
from collections import Counter, defaultdict
from ntpath import join
from torch.utils.data import SubsetRandomSampler
import numpy as np
import pandas as pd
from pytest import param
from sqlalchemy import column, desc
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
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
from text_model import ProductDescpMannual

pd.options.display.max_colwidth = 400
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 40)
plt.rc('axes', titlesize=12)
final_layer_num = 24

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

'''IMAGE MODEL'''
prod_dum = CleanData()
class_dict = prod_dum.major_map_encoder.keys()
classes = list(class_dict)
class_values = prod_dum.major_map_encoder.values()
class_encoder = prod_dum.major_map_encoder
class_decoder = prod_dum.major_map_decoder

ImageClassifier = models.resnet50(pretrained=True)
for i, param in enumerate(ImageClassifier.parameters(), start=1):
    param.requires_grad=False

def num_out(major=True, cutoff_lim = 50):
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
        nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=True), nn.Flatten(), nn.Linear(96, 128), nn.Tanh(),nn.Linear(128, final_layer_num)) # !!!!!!!!!!!!!!!!!!!!!!
    
    def forward(self, inp):
        x = self.main(inp)
        return x


'''COMBINED MODEL'''
'''Combined Model Classifier'''
class CombinedModel(nn.Module):
    def __init__(self, num_classes, input_size=768, decoder=class_decoder) -> None:
        super(CombinedModel, self).__init__()
        self.image_classifier = ImageClassifier
        self.text_classifier = DescriptionClassifier(input_size=input_size)
        self.main = nn.Sequential(nn.Linear(2*final_layer_num, num_classes))
        self.decoder = decoder

    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features) # !!!!!!!!!!!!!!!!!!!!!!
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

    def predict(self, image, txt):
        with torch.no_grad():
            x = self.forward(image_features=image, text_features=txt) # !!!!!!!!!!!!!!!!!!!!!!
            return x
    
    def predict_proba(self, image, txt):
        with torch.no_grad():
            x = self.forward(image_features=image, text_features=txt)
            return torch.softmax(x, dim=1)
    
    def predict_class(self, image, txt):
        with torch.no_grad():
            x = self.forward(image_features=image, text_features=txt)
            return self.decoder[int(torch.argmax(x, dim=1))]

'''IMAGE PROCESSOR FOR UPLOAD ONTO API'''
class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) # is this right?
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(self.repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def repeat_channel(x):
            return x.repeat(3, 1, 1)

    def __call__(self, image):
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)
        # Add a dimension to the image
        image = image[None, :, :, :]
        return image

'''TEXT PROCESSOR FOR UPLOAD ONTO API'''
def text_process(txt, limit_it=True, vocab_limit=30000):
    prod_class = ProductDescpMannual()
    word_encoder, word_decoder, _ = prod_class.vocab_encoder(col='product_description', limit_it=limit_it, vocab_limit=vocab_limit)
    stop_words = set(stopwords.words('english'))
    word_lemmitizer = WordNetLemmatizer()
    punct_re = re.compile(r'[^A-Za-z \n-]') #[^A-Za-z0-9 .]
    dum_ls = txt.split()
    # self.product_frame[col] = self.product_frame[col].copy().str.replace(punct_re, '')
    dum_ls = [re.sub(punct_re, '', i) for i in dum_ls]
    dum_ls = [i for i in dum_ls if i not in  stop_words]
    dum_ls = [i.replace('\n', ' ') for i in dum_ls]
    dum_ls = [i.strip('\-') for i in dum_ls]
    dum_ls = [word_lemmitizer.lemmatize(i) for i in dum_ls]
    dum_ls = [contractions.fix(j) for j in dum_ls]
    dum_ls = [i for i in dum_ls if i.isalpha] # and len(i)>2
    dum_ls = [i.lower() for i in dum_ls]
    new_txt = ' '.join(dum_ls)
    return new_txt

class TextProcessor:
    def __init__(self, max_length=30):
        prod = ProductDescpMannual()
        self.full_word_ls, _ = prod.clean_prod()
        self.product_description_counter = Counter(self.full_word_ls)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()

    def __call__(self, txt, min_count=2):
        processed_txt = text_process(txt=txt)
        pre_ls = processed_txt.split()
        pre_ls = [i for i in pre_ls if self.product_description_counter[i]>min_count]
        newly_processed_txt = ' '.join(pre_ls)
        bert_encoded = self.tokenizer.batch_encode_plus([newly_processed_txt], max_length=self.max_length, padding='max_length', truncation=True)
        bert_encoded = {key: torch.LongTensor(value) for key, value in bert_encoded.items()}
        with torch.no_grad():
            new_description = self.model(**bert_encoded).last_hidden_state.swapaxes(1,2)
        # new_description = new_description.squeeze(0)
        print(new_description)
        print('Its size: ', new_description.size())
        return new_description
        

'''COMBINED MODEL WEIGHTS'''
combined_model=CombinedModel(num_classes=len(class_decoder))
combined_model.load_state_dict(torch.load('image_text_combined.pt', map_location='cpu'))

app = FastAPI()
image_processor = ImageProcessor()
text_processor = TextProcessor()


@app.post('/combined')
def product_upload(img: UploadFile=File(...), description: str=Form(...)):
    uploaded_image = Image.open(img.file)
    processed_img = image_processor(uploaded_image)
    processed_txt = text_processor(description)
    print(processed_txt)
    predicted_raw = combined_model.predict(processed_img, processed_txt) # !!!!!!!!!!!!!!!!!!!!!!
    predicted_probabilities = combined_model.predict_proba(processed_img, processed_txt)
    predicted_class = combined_model.predict_class(processed_img, processed_txt)
    print('Predicted Raw Values:\n', predicted_raw)
    print('Predicted Probabilities:\n', predicted_probabilities)
    print('Predicted Class: ', predicted_class)
    return JSONResponse(status_code=200, content={'prediction_raw': predicted_raw.tolist(), 'prediction_porba': predicted_probabilities.tolist(), 'predicted_class': predicted_class})

# app = FastAPI()
if __name__ == '__main__':
    print('Length of decoder: ', len(class_decoder))
    uvicorn.run('model_api:app', host='0.0.0.0', port=8080)