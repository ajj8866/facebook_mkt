from math import ceil
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
from sklearn.preprocessing import LabelEncoder
from matplotlib.gridspec import GridSpec

class CleanData:
    def __init__(self, tab_names = 'products_2') -> None:
        self.tab_names = tab_names
        self.con = aws_rds()[1]
        if 'test_file' not in os.listdir():
            os.mkdir(Path(Path.cwd(), 'test_file'))
        if isinstance(tab_names, str):
            self.df = pd.read_sql_table(table_name=tab_names, con=self.con)
            if 'price' in self.df.columns:
                self.df.dropna(inplace=True)
                self.df['price'] = self.df['price'].str.replace('£', '').astype(np.float32)
                self.df = self.df[np.round(self.df['price']) != 0]
        else:
            self.table_dict = {}
            for table in tab_names:
                self.table_dict[table] = pd.read_sql_table(table_name=table, con= self.con)
                self.table_dict[table].dropna(inplace = True)
                if 'price' in self.table_dict[table].columns:
                    self.table_dict[table]['price'] = self.table_dict[table][self.table_dict[table]['price'] != 'N/A'.strip()]['price']
                    print(self.table_dict[table].head())
                    self.table_dict[table]['price'] = self.table_dict[table]['price'].str.replace(',', '').str.strip('£').str.strip(' ').astype(np.float32)
                    self.table_dict[table] = self.table_dict[table][np.round(self.table_dict[table]['price']) != 0]
    
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
            ex_writer = pd.ExcelWriter(f'test_file/{i}.xlsx', engine='xlsxwriter')
            with ex_writer as writer:
                j.to_excel(writer, sheet_name=i)
    
    def expand_category(self, df = 'combined'):
        self.major_encoder = LabelEncoder()
        self.minor_encoder = LabelEncoder()
        self.table_dict[df]['major_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[0])
        self.table_dict[df]['minor_category'] = self.table_dict[df]['category'].str.split('/').apply(lambda i: i[1])
        self.table_dict[df]['major_category_encoded'] = self.major_encoder.fit_transform(self.table_dict[df]['major_category'])
        self.table_dict[df]['minor_category_encoded'] = self.minor_encoder.fit_transform(self.table_dict[df]['minor_category'])
        return self.table_dict[df]
    
    def sum_by_cat(self, df= 'combined', quant = 0.95):
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

    def trim_data(self, df= 'combined', quant = 0.95):
        self.table_dict[df] = self.table_dict[df][self.table_dict[df]['price'] > self.table_dict[df]['price'].quantile([quant])]

    @classmethod
    def allTables(cls):
        engine = aws_rds()[0]
        inspector = inspect(engine)
        return cls(tab_names = inspector.get_table_names())
    
##########################################################################################################################################
##########################################################################################################################################

if __name__ == '__main__':
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    data_class = CleanData.allTables()
    print(data_class)
    data_class.try_merge(df_list=['products', 'products_2'])
    data_class.get_na_vals(df='combined')
    data_class.expand_category()
    data_class.sum_by_cat()
    data_class.to_excel()
    print('#'*20)
    print(data_class.table_dict['combined'].head())
    print('#'*20)

