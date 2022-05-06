from sqlalchemy import create_engine, MetaData, inspect
import boto3
from aws_tool import *
import numpy as np
import pandas as pd
import xlsxwriter
import os
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from clean_tabular import CleanData
from skimage.io import imread, imshow
from skimage import io
from skimage import img_as_float
from skimage.transform import rescale, resize
from PIL import Image

class CleanImages(CleanData):
    def __init__(self, tab_names='images') -> None:
        super().__init__(tab_names)
        print(self.tab_names)
        print(self.df.head())
        self.csv_df = None

    def img_clean_pil(self, size = 512, mode = 'RGB'):
        image_re = re.compile(r'(.*)\.jpg')
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt', 'images'))
        t = 0
        for i in os.listdir():
            if re.findall(image_re, i) != []:
                temp_image = Image.open(i)
                black_back = Image.new(size=(size, size), mode=temp_image.mode) #, mode=mode
                curr_size = temp_image.size
                max_dim = max(temp_image.size)
                scale_fact = size / max_dim
                resized_image_dim = (int(scale_fact*curr_size[0]), int(scale_fact*curr_size[1]))
                updated_image = temp_image.resize(resized_image_dim)
                black_back.paste(updated_image, ((size- resized_image_dim[0])//2, (size- resized_image_dim[1])//2))
                black_back = black_back.convert(mode)
                t += 1
                print(t)
                black_back.save(i)
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))

    def img_clean_sk(self, normalize = False):
        image_re = re.compile(r'(.*)\.jpg')
        img = []
        img_dim_list = []
        img_id = []
        image_array = []
        img_channels = []
        img_num_features = []
        img_mode = []
        os.chdir(Path(Path.cwd(), 'images'))
        for im in os.listdir():
            if re.findall(image_re, im) != []:
                img.append(im)
                image = io.imread(im)
                if normalize == True:
                    image = img_as_float(image)
                img_id.append(re.search(image_re, im).group(1))
                image_array.append(image)
                img_dim_list.append(image.shape)
                if len(image.shape) == 3:
                    print(im)
                    img_num_features.append(image.shape[2])
                else:
                    img_num_features.append(1)
                img_channels.append(len(image.shape))
                img_mode.append(Image.open(im).mode)
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))
        self.image_frame = pd.DataFrame(data={'image_id': img_id, 'image': img,'image_array': image_array,'image_shape': img_dim_list, 'mode': img_mode})
        print(self.image_frame.head())
        return self.image_frame
    
    def to_excel(self, df):
        df.to_excel(Path(Path.cwd(), 'test_file','Cleaned_Images.xlsx'), sheet_name = 'images')

    def merge_images(self):
        self.df.rename({'id': 'image_id', 'product_id': 'id'}, axis=1, inplace=True)
        self.final_df = self.image_frame.merge(self.df, on='image_id', how='inner', validate='one_to_many')
        print(self.final_df.head())
        return self.final_df

    def total_clean(self, normalize=True, mode = 'L', size = 128):
        self.img_clean_pil(mode=mode, size=size)
        self.img_clean_sk(normalize=normalize)
        self.merge_images()
        return self.final_df
    
    def describe_data(self, df):
        print('\n')
        print('Data frame columnn information')
        print(df.info())
        print('\n')
        print('#'*20)
        print('Dataframe statistical metrics')
        #print(df.describe())
        print('#'*20)
        print('Array and shape')
        print(df['image_shape'].unique())
        print(df['image_shape'].value_counts())

if __name__ == '__main__':
    print(os.getcwd())
    pd.set_option('display.max_colwidth', 400)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    cl = CleanImages()
    print(cl)
    cl.total_clean()
    cl.to_excel(cl.final_df)
    print(cl.final_df.head())
    cl.describe_data(cl.final_df)
