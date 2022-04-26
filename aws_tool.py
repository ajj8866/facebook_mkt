
import sys
import time
import pandas as pd
import re
from sqlalchemy import create_engine
import boto3
from pathlib import Path
import xlsxwriter 
import os


def aws_rds():
    DATABASE_TYPE = 'postgresql'
    DBAPI = 'psycopg2'
    ENDPOINT = 'products.c8k7he1p0ynz.us-east-1.rds.amazonaws.com'
    USER = 'postgres'
    PASSWORD = 'aicore2022!'
    PORT = 5432
    DATABASE = 'postgres'
    engine = create_engine(f'{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}')
    return engine, engine.connect()

def aws_s3_upload(upload_file, bucket_file_alis, bucket_name):
    s3_res = boto3.resource('s3')
    resp = s3_res.upload_file(upload_file, bucket_file_alis, bucket_name)

def aws_s3_download(s3_url):
    s3_res = boto3.resource('s3')


def ls_buckets():
    s3 = boto3.resource('s3')
    buck_ls = []
    for i in s3.buckets.all():
        buck_ls.append(i)
    return buck_ls

def aws_s3_upload_folder(bucket_name = 'datapipelines3fx', path = Path(Path.cwd(), 'raw_data', 'images')):
    sess = boto3.client('s3')
    for i in os.listdir(path):
        try:
            home = Path(Path.home(), 'Downloads', 'AICore', 'Datapipe')
            os.chdir(Path(Path.home(), 'Downloads', 'AICore','Datapipe','raw_data', 'images'))
            sess.upload_file(i, bucket_name,i)
            os.chdir(home)
        except:
            os.chdir(Path(Path.cwd(), 'raw_data', 'images'))
            sess.upload_file(i, bucket_name, i)
            os.chdir(Path.cwd().parents[1])