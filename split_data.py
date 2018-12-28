# coding: utf-8


import os
import shutil
import numpy as np
import pandas as pd


data_path = './data'
train_data_path = './data/train'
valid_data_path = './data/valid'


# 移除之前已经存在的文件夹
if os.path.exists(data_path):
    shutil.rmtree(data_path)


# 创建新的文件夹
df = pd.read_csv('./driver_imgs_list.csv')
classname_list = df['classname'].unique()
for classname in classname_list:
    os.makedirs(os.path.join(train_data_path, classname))
    os.makedirs(os.path.join(valid_data_path, classname))


# 按司机 ID 划分数据集
driver_list = df['subject'].unique()
np.random.shuffle(driver_list)
train_driver_list = driver_list[0:int(len(driver_list) * 0.8)]
valid_driver_list = driver_list[int(len(driver_list) * 0.8):-1]
train_df = df[df['subject'].isin(train_driver_list)]
valid_df = df[df['subject'].isin(valid_driver_list)]


def create_train_data(class_name, df):
    for row in df[df['classname'] == class_name].itertuples():
        src = os.path.join('./imgs/train', getattr(row, 'classname'), getattr(row, 'img'))
        dst = os.path.join(train_data_path, getattr(row, 'classname'), getattr(row, 'img'))
        shutil.copy(src, dst)


def create_valid_data(class_name, df):
    for row in df[df['classname'] == class_name].itertuples():
        src = os.path.join('./imgs/train', getattr(row, 'classname'), getattr(row, 'img'))
        dst = os.path.join(valid_data_path, getattr(row, 'classname'), getattr(row, 'img'))
        shutil.copy(src, dst)


if __name__ == '__main__':

    for classname in classname_list:
        create_train_data(classname, train_df)
        create_valid_data(classname, valid_df)

