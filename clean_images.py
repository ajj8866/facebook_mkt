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
        img_num_features = []
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt', 'images'))
        for i in os.listdir():
            if re.findall(image_re, i) != []:
                temp_image = Image.open(i)
                black_back = Image.new(size=(size, size), mode=mode)
                curr_size = temp_image.size
                max_dim = max(temp_image.size)
                scale_fact = size / max_dim
                resized_image_dim = (int(scale_fact*curr_size[0]), int(scale_fact*curr_size[1]))
                updated_image = temp_image.resize(resized_image_dim)
                black_back.paste(updated_image, ((size- resized_image_dim[0])//2, (size- resized_image_dim[1])//2))
                if black_back.mode == 'L':
                    black_back = black_back.convert('RGB')
                print(black_back.mode)
                black_back.save(i)
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))

    def img_clean_sk(self, normalize = False):
        image_re = re.compile(r'(.*)\.jpg')
        img_dim_list = []
        img_id = []
        image_array = []
        img_channels = []
        img_num_features = []
        img_mode = []
        os.chdir(Path(Path.cwd(), 'images'))
        for im in os.listdir():
            if re.findall(image_re, im) != []:
                image = io.imread(im)
                if normalize == True:
                    image = img_as_float(image)
                img_id.append(re.search(image_re, im).group(1))
                image_array.append(image)
                img_dim_list.append(image.shape)
                if len(image.shape) == 3:
                    img_num_features.append(image.shape[2])
                else:
                    img_num_features.append(1)
                img_channels.append(len(image.shape))
                img_mode.append(Image.open(im).mode)

        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))
        image_frame = pd.DataFrame(data={'Image_ID': img_id, 'Image_Array': image_array,'Image_Shape': img_dim_list, 'Mode': img_mode})
        print(image_frame.head())
        return image_frame

    def pd_to_csv(self):
        self.csc_df = pd.read_csv('Images/Links.csv')
        print(len(self.csc_df))

    
if __name__ == '__main__':
    print(os.getcwd())
    pd.set_option('display.max_colwidth', 400)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    cl = CleanImages()
    print(cl)
    print(cl.df.head())
    cl.pd_to_csv()
    cl.img_clean_pil()
    cl.img_clean_sk()