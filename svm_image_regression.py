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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from clean_images import CleanImages
from clean_tabular import CleanData
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 400)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.max_rows', 40)
    img_size = 224
    '''Getting images dataframe'''
    image_class = CleanImages()
    image_df = image_class.total_clean(size=img_size, mode='RGB')

    '''Getting product data'''
    product_class = CleanData(tab_names=['Products'])
    product_class.get_na_vals(df='Products')
    products_df = product_class.table_dict['Products']

    '''Dataframe diagnostics'''
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

    ### 

    '''Merging dataframes'''
    merged_df = image_df.merge(products_df, left_on='id', right_on='id')
    print(merged_df.head())
    print(merged_df.info())
    merged_df.to_excel(Path(Path.cwd(), 'data_files', 'Final_Data_new.xlsx'))
    print(len(merged_df))

    ''' Dataset Preprocessing'''
    filtered_df = merged_df.loc[:, ['edge_array','image_id', 'image_array', 'minor_category', 'major_category']].copy()
    num_obs = 120
    filtered_df = filtered_df.iloc[:num_obs]
    X = filtered_df['edge_array'].copy() #filtered_df['image_array'].copy()
    y = filtered_df['major_category'].copy()
    X = np.vstack(X.values)
    X = X.reshape((-1, img_size*img_size))    
    category_list = list(set(y))
    print('\n')
    print('List of possible categories for classification')
    print('\n'.join([f'{i}) {j}' for i, j in enumerate(category_list, start=1)]))
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    '''Setting up GridSearchCV'''
    param_grid = {'C': [0.001, 0.01, 0.1, 1]}
    svc = SVC(gamma='auto', kernel='rbf')
    model = GridSearchCV(estimator=svc, param_grid=param_grid)
    start_time = time.time()
    model.fit(X_train, y_train)
    print(model.best_params_)
    total_secs = time.time() - start_time
    if time.time() - start_time >= 60:
        mins = (total_secs)//60
        secs = ((total_secs/60)-mins)*60
        print(f'Time taken to train model comprising of {num_obs} observations: ', f'{int(mins)} minutes and {int(secs)} seconds')
    else:
        print(f'Time taken to train model comprising of {num_obs} observations: ', f'{round(total_secs, 2)} seconds taken')
    y_pred = model.predict(X_test)
    print(product_class.major_map_decoder.values())
    '''Model Prediction Evaluation'''
    print('Accuracy score: ', f'{accuracy_score(y_test, y_pred)*100:.2f}%')
    confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=list(product_class.major_map_decoder.values())), annot=True, xticklabels=True)
    plt.show()

    