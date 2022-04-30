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
import sklearn
from sklearn.svm import SVC
from clean_images import CleanImages
from clean_tabular import CleanData
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 400)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)

    # Getting images dataframe
    image_class = CleanImages()
    image_df = image_class.total_clean()

    # Getting product data
    product_class = CleanData(tab_names=['products', 'products_2'])
    product_class.try_merge(['products', 'products_2'])
    product_class.get_na_vals(df='combined')
    products_df = product_class.expand_category()

    # Dataframe diagnostics
    print('\n')
    print('Image Dataframe info')
    print(image_df.info())
    print(image_df.head())
    print(image_df.columns)

    print('\n')
    print('Products Dataframe info')
    print(products_df.info())
    print(products_df.head())
    print(products_df.columns)

    # Merging dataframes
    merged_df = image_df.merge(products_df, left_on='id', right_on='id')
    print(merged_df.head())
    print(merged_df.info())
    merged_df.to_excel(Path(Path.cwd(), 'test_file', 'Final_Data.xlsx'))
    print(len(merged_df))



