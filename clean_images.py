import imp
from sqlalchemy import create_engine, MetaData, inspect
import numpy as np
import pandas as pd
import xlsxwriter
import os
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from clean_tabular import CleanData
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.io import imread, imshow
from skimage import io
import json
from skimage import img_as_float
import re
from skimage.transform import rescale, resize
from itertools import product
from PIL import Image

class CleanImages(CleanData):
    def __init__(self, tab_names=['Images']) -> None:
        super().__init__(tab_names)
        self.df = self.table_dict[tab_names[0]].copy()
        self.csv_df = None

    def img_clean_pil(self, size = 512, mode = 'RGB'):
        image_re = re.compile(r'(.*)\.jpg')
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt', 'images'))
        # os.chdir(Path(Path.cwd(), 'images'))
        t = 0
        for i in os.listdir():
            if re.findall(image_re, i) != []:
                try:
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
                    black_back.save(i)
                except Exception:
                    print(i)
                    with open('invalid_file.json', 'w') as wrong_form:
                        json.dump(i, wrong_form)
                    os.remove(i)
                    pass
        print(t)
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
                print(image.shape)
                img_dim_list.append(image.shape)
                if len(image.shape) == 3:
                    img_num_features.append(image.shape[2])
                else:
                    print(im)
                    img_num_features.append(1)
                img_channels.append(len(image.shape))
                img_mode.append(Image.open(im).mode)
        os.chdir(Path(Path.home(), 'Downloads', 'AICore', 'facebook_mkt'))
        self.image_frame = pd.DataFrame(data={'image_id': img_id, 'image': img,'image_array': image_array,'image_shape': img_dim_list, 'mode': img_mode})
        print(self.image_frame.head())
        return self.image_frame
    
    def to_excel(self, df):
        df.to_excel(Path(Path.cwd(), 'data_files','Cleaned_Images.xlsx'), sheet_name = 'images')

    def merge_images(self):
        self.df.rename({'id': 'image_id', 'product_id': 'id'}, axis=1, inplace=True)
        self.final_df = self.image_frame.merge(self.df, on='image_id', how='inner', validate='one_to_many')
        print(self.final_df.head())
        return self.final_df
    
    def edge_detect(self):
        try:
            self.image_frame['edge_array'] = self.image_frame['image_array'].copy().apply(lambda i: sobel(rgb2gray(i)))
        except: 
            self.image_frame['edge_array'] = self.image_frame['image_array'].copy().apply(lambda i: sobel(i))
        return self.image_frame


    def total_clean(self, normalize=False, mode = 'RGB', size = 224):
        self.img_clean_pil(mode=mode, size=size)
        self.img_clean_sk(normalize=normalize)
        self.edge_detect()
        self.merge_images()
        return self.final_df
    
    def show_random_images(self, col, size, fig_height= 15, fig_width=10):
        grid = GridSpec(nrows = size, ncols = size)
        fig = plt.figure(figsize=(fig_height, fig_width))
        for i, j in product(range(size), range(size)):
            fig.add_subplot(grid[i, j]).imshow(self.final_df[col].iloc[np.random.randint(low=0, high=len(self.final_df)-1)])
        plt.show()

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

class MergedData:
    def __init__(self):
        img_class = CleanImages()
        prod_class = CleanData(tab_names=['Products'])
        self.major_map_encoder = prod_class.major_map_encoder
        self.major_map_decoder = prod_class.major_map_decoder
        self.prod_frame = prod_class.table_dict['Products'].copy()
        self.img_df = img_class.total_clean()
        self.merged_frame = self.img_df.merge(self.prod_frame, left_on='id', right_on='id')
    
    def to_pickle(self):
        self.merged_frame.to_pickle(Path(Path.cwd(), 'merged_data.pkl'))
    
    def get_val_counts(self):
        return {'products': self.prod_frame, 'images': self.img_df, 'all': self.merged_frame}

if __name__ == '__main__':
    print(os.getcwd())
    pd.set_option('display.max_colwidth', 400)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    cl = CleanImages()
    cl.total_clean()
    cl.to_excel(cl.final_df)
    print(cl.final_df.head())
    cl.describe_data(cl.final_df)
    print(cl)
    cl.show_random_images(col='edge_array', size=4)
    # merged_class = MergedData()
    # merged_class.get_val_counts()
    # merged_class.to_pickle()
