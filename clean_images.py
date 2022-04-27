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


class CleanImages(CleanData):
    def __init__(self, tab_names='images') -> None:
        super().__init__(tab_names)
        print(self.tab_names)
        print(self.df.head())
        self.csv_df = None


    def img_shape(self):
        image_re = re.compile(r'(.*)\.jpg')
        img_dim_list = []
        img_id = []
        os.chdir(Path(Path.cwd(), 'images'))
        for im in os.listdir():
            if re.findall(image_re, im) != []:
                image = imread(os.path.abspath(im))
                #print(image.shape)
                img_id.append(re.search(image_re, im).group(1))
                img_dim_list.append(image.shape)
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))
        image_frame = pd.DataFrame(data={'Image_ID': img_id, 'Image_Shape': img_dim_list})
        image_frame.loc[:, 'Num_Channels'] = image_frame['Image_Shape'].apply(lambda i: len(i))
        print(image_frame['Num_Channels'].unique())
        print(image_frame['Num_Channels'].value_counts())
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
    cl.img_shape()