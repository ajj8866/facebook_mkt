from math import ceil
from sqlalchemy import create_engine, MetaData, inspect
import numpy as np
import pandas as pd
import xlsxwriter
import os
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from matplotlib.gridspec import GridSpec

class CleanData:
    def __init__(self, tab_names = ['Products']) -> None:
        self.tab_names = tab_names
        maj_unique_cats = ['Home & Garden ', 'Baby & Kids Stuff ', 'DIY Tools & Materials ', 'Music, Films, Books & Games ', 'Phones, Mobile Phones & Telecoms ', 'Clothes, Footwear & Accessories ', 'Other Goods ', 'Health & Beauty ', 'Sports, Leisure & Travel ', 'Appliances ', 'Computers & Software ','Office Furniture & Equipment ', 'Video Games & Consoles ']
        self.major_map_decoder = dict(enumerate(maj_unique_cats))
        self.major_map_encoder = {val: key for key, val in self.major_map_decoder.items()}
        print(self.major_map_decoder)
        print(self.major_map_encoder)
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
                self.expand_category(df=table)
                print(self.table_dict[table].head())

    
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
    
    def expand_category(self, df = 'Products'):
        self.major_encoder = LabelEncoder()
        self.minor_encoder = LabelEncoder()
        self.table_dict[df]['major_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[0])
        self.table_dict[df]['minor_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[1])
        self.table_dict[df] = self.table_dict[df][self.table_dict[df]['major_category'] != 'N'.strip()]
        print('Encoder', self.major_map_encoder)
        self.table_dict[df]['major_category_encoded'] = self.table_dict[df]['major_category'].map(self.major_map_encoder)
        self.table_dict[df]['minor_category_encoded'] = self.minor_encoder.fit_transform(self.table_dict[df]['minor_category'])
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

if __name__ == '__main__':
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    data_class = CleanData(tab_names=['Products'])
    print(data_class)
    data_class.get_na_vals(df='Products')
    # data_class.sum_by_cat()
    # print(data_class.cat_set())