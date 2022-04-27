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
        self.csc_df = None


    def img_shape(self):
        image_re = re.compile(r'.*\.jpg')
        img_dim_list = []
        for im in os.listdir(Path(Path.cwd(), 'images')):
            if re.findall(image_re, im) !=[]:
                print(im)
                image = imread(im)
                img_dim_list.append(image.shape)
        print(img_dim_list)



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