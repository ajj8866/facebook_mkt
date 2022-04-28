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
from skimage.transform import rescale, resize

class CleanImages(CleanData):
    def __init__(self, tab_names='images') -> None:
        super().__init__(tab_names)
        print(self.tab_names)
        print(self.df.head())
        self.csv_df = None


    def img_shape(self, preserve_orig = False, width = 512, height = 512):
        image_re = re.compile(r'(.*)\.jpg')
        img_dim_list = []
        img_id = []
        image_array = []
        img_width = []
        img_height = []
        img_channels = []
        img_num_features = []
        os.chdir(Path(Path.cwd(), 'images'))
        for im in os.listdir():
            if re.findall(image_re, im) != []:
                image = io.imread(im)
                image = resize(image, output_shape=(width,height), mode = 'constant', cval=0, preserve_range=preserve_orig)
                print(len(image.shape))
                image = [np.expand_dims(image, axis = -1) if len(image.shape) == 2 else image][0]
                img_id.append(re.search(image_re, im).group(1))
                image_array.append(image)
                img_dim_list.append(image.shape)
                img_width.append(image.shape[0])
                img_height.append(image.shape[1])
                img_num_features.append(image.shape[2])
                img_channels.append(len(image.shape))
                if image.shape[2] == 1:
                    print(image)
                    image = image.reshape(image.shape[0], image.shape[1])
                    plt.imsave(fname=f'{re.search(image_re, im).group(1)}.jpg', arr=image)
                else:
                    plt.imsave(fname=f'{re.search(image_re, im).group(1)}.jpg', arr=image)
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))
        image_frame = pd.DataFrame(data={'Image_ID': img_id, 'Image_Array': image_array,'Image_Shape': img_dim_list, 'Image_Width': img_width, 'Image_Height': img_height, 'Num_Channels': img_channels})
        print('Maximum width of images: ', np.max(image_frame['Image_Width']))
        print('Maximum height: ', np.max(img_height))
        print('Minimum width of images: ', np.min(image_frame['Image_Width']))
        print('Minimum height: ', np.min(img_height))
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