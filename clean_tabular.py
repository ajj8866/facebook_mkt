import enum
from collections import Counter, defaultdict
from ntpath import join
import numpy as np
import pandas as pd
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

class CleanData:
    def __init__(self, level=1, tab_names = ['Products']) -> None:
        self.tab_names = tab_names
        maj_unique_cats = ['Home & Garden ', 'Baby & Kids Stuff ', 'DIY Tools & Materials ', 'Music, Films, Books & Games ', 'Phones, Mobile Phones & Telecoms ', 'Clothes, Footwear & Accessories ', 'Other Goods ', 'Health & Beauty ', 'Sports, Leisure & Travel ', 'Appliances ', 'Computers & Software ','Office Furniture & Equipment ', 'Video Games & Consoles ']
        self.major_map_decoder = dict(enumerate(maj_unique_cats))
        self.major_map_encoder = {val: key for key, val in self.major_map_decoder.items()}
        # print(self.major_map_decoder)
        # print(self.major_map_encoder)
        if 'data_files' not in os.listdir():
            os.mkdir(Path(Path.cwd(), 'data_files'))
        self.table_dict = {}
        for table in tab_names:
            self.table_dict[table] = pd.read_json(Path(Path.cwd(),'data_files', table+'.json'))
            self.table_dict[table].dropna(inplace = True)
            if 'price' in self.table_dict[table].columns:
                self.table_dict[table]['price'] = self.table_dict[table][self.table_dict[table]['price'] != 'N/A'.strip()]['price']
                self.table_dict[table]['price'] = self.table_dict[table]['price'].str.replace(',', '').str.strip('£').str.strip(' ').astype(np.float32)
                self.table_dict[table] = self.table_dict[table][np.round(self.table_dict[table]['price']) != 0]
            if 'category' in self.table_dict[table].columns:
                self.expand_category(df=table, level=level)
                # print(self.table_dict[table].head())
                print(len(self.table_dict[table]['minor_category_encoded'].value_counts()))
    
    
    def try_merge(self, df_list):
        '''
        Combines dataframes passed in into a single dataframe

        Parameters:
        df_list: Must contain dataframes within self.table_dict passed in as a list
        '''
        if isinstance(self.tab_names, str):
            print('Method not valid when class instantiated with tab_names as type string')
        else:
            self.new_df = pd.DataFrame(columns = self.table_dict[df_list[0]].columns)
            for i in df_list:
                self.new_df = pd.concat([self.new_df, self.table_dict[i]], axis=0)
        self.table_dict['combined'] = self.new_df
        self.table_dict['combined'].dropna(inplace=True)
        print(self.table_dict['combined'].head())
        return self.table_dict['combined']
    
    def get_na_vals(self, df):
        print(f'The following NA values exist if dataframe {df}')
        return self.table_dict[df][self.table_dict[df].isna().any(axis=1)]

    def __repr__(self) -> str:
        if isinstance(self.tab_names, str):
            print(self.df.columns)
            print('\nTable Name: ', self.tab_names, 'With columns:')
            return ' | '.join(self.df.columns)
        else:
            print('\n')
            print('Total of ', f'{len(self.table_dict)} tables')
            return '\n'.join([f'Table Name: {i}: \n' f'Columns | {" | ".join(j.columns)} \n' for i, j in self.table_dict.items()])

    def to_excel(self):
        for i, j in self.table_dict.items():
            ex_writer = pd.ExcelWriter(f'data_files/{i}.xlsx', engine='xlsxwriter')
            with ex_writer as writer:
                j.to_excel(writer, sheet_name=i)
    
    def cat_set(self, df = 'Products',cat_col = 'major_category'):
        return self.table_dict[df][cat_col].nunique()
    
    def expand_category(self, df = 'Products', level=1):
        self.minor_encoder = LabelEncoder()
        self.table_dict[df]['major_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[0])
        self.table_dict[df]['minor_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[level])
        self.table_dict[df] = self.table_dict[df][self.table_dict[df]['major_category'] != 'N'.strip()]
        print('Encoder', self.major_map_encoder)
        self.table_dict[df]['major_category_encoded'] = self.table_dict[df]['major_category'].map(self.major_map_encoder)
        self.table_dict[df]['minor_category_encoded'] = self.minor_encoder.fit_transform(self.table_dict[df]['minor_category'].sort_values(key=lambda i: i.str.lower()))
        return self.table_dict[df]
    
    def inverse_transform(self, input_array, major_minor = 'minor'):
        category_dict = {'major': self.major_encoder, 'minor': self.minor_encoder}
        try:
            return category_dict[major_minor].inverse_transform(input_array)
        except TypeError:
            return category_dict[major_minor].inverse_transform(input_array.numpy())
    
    
    def sum_by_cat(self, df= 'Products', quant = 0.95):
        data = self.expand_category(df)
        major = data.groupby('major_category')['price'].describe()
        print('Price Statistics Grouped by Major Category')
        print(major)
        major_cat_list = major.index.tolist()
        #sns.boxplot(data=data, x = 'major_category', y = 'price')
        products_df = data.loc[:, ['major_category', 'minor_category', 'price']]
        for i in major_cat_list:
            prod_plot = products_df.loc[products_df['major_category'] == i]
            print(prod_plot['price'].quantile([quant]))
            print(type(prod_plot['price'].quantile([quant][0])))
            print('Number of observations with price more than the 99th quantile: ', len(prod_plot[prod_plot['price'] > prod_plot['price'].quantile([quant][0])]))
            # sns.boxplot(data=prod_plot, x='major_category', y='price')
            # plt.show()
            sns.boxplot(data=prod_plot[prod_plot['price']<prod_plot['price'].quantile([quant][0])], x = 'major_category', y = 'price')
            plt.show()

    def trim_data(self, df= 'Products', quant = 0.95):
        self.table_dict[df] = self.table_dict[df][self.table_dict[df]['price'] > self.table_dict[df]['price'].quantile([quant])]
        return self.table_dict[df]

    @classmethod
    def allTables(cls):
        json_list = []
        json_regex = re.compile(r'(.*).json$')
        for i in os.listdir(Path(Path.cwd(), 'data_files')):
            if re.search(json_regex, i) is not None:
                json_list.append(re.search(json_regex, i).group(1))
        print(json_list)
        return cls(tab_names = json_list)
    
##########################################################################################################################################
##########################################################################################################################################


class CleanImages(CleanData):
    def __init__(self, level=1,tab_names=['Images']) -> None:
        super().__init__(level=level, tab_names=tab_names)
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
        #self.edge_detect()
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


##########################################################################################################################################
##########################################################################################################################################

class MergedData:
    def __init__(self):
        img_class = CleanImages()
        prod_class = CleanData(tab_names=['Products'])
        self.major_map_encoder = prod_class.major_map_encoder
        self.major_map_decoder = prod_class.major_map_decoder
        self.prod_frame = prod_class.table_dict['Products'].copy()
        self.img_df = img_class.total_clean()
        self.merged_frame = self.img_df.merge(self.prod_frame, left_on='id', right_on='id')
        # self.merged_frame = self.merged_frame.loc[:, ['image_id', 'product_description']]
    
    def get_val_counts(self):
        return {'products': self.prod_frame, 'images': self.img_df, 'all': self.merged_frame}



if __name__ == '__main__':
    pass
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    # merged = MergedData()
    prod = CleanData(level=1)

