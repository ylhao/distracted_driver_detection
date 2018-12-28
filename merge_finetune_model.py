# coding: utf-8


import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Lambda
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import adam, Adam
from sklearn.utils import shuffle
from keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')  # 服务器端使用 matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.utils import plot_model


# def plot_acc(history):
#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('Acc')
#     plt.plot(history.epoch, np.array(history.history['acc']), label='Train Acc')
#     plt.plot(history.epoch, np.array(history.history['val_acc']), label = 'Val Acc')
#     plt.legend()
#     plt.ylim([0, 1])
#     plt.savefig('acc.png')


# def plot_loss(history):
#     plt.figure()
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss')
#     plt.plot(history.epoch, np.array(history.history['val_loss']), label = 'Val Loss')
#     plt.legend()
#     plt.ylim([0, 5])
#     plt.savefig('loss.png')


def mergeFinetuneModel():
    X_train = []
    X_valid = []
    filenames = ['inceptionv3-finetune-output.hdf5', 'resnet50-finetune-output.hdf5', 'xception-finetune-output.hdf5']
    for filename in filenames:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['X_train']))
            X_valid.append(np.array(h['X_val']))
            y_train = np.array(h['y_train'])
            y_valid = np.array(h['y_val'])
    X_train = np.concatenate(X_train, axis=1)
    X_valid = np.concatenate(X_valid, axis=1)

    # check
    print('X_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)

    X_train, y_train = shuffle(X_train, y_train)
    y_train = to_categorical(y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    y_valid = to_categorical(y_valid)

    # check
    print('X_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)

    inputs = Input(X_train.shape[1:])
    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs, predictions)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    check_point = ModelCheckpoint(filepath='./model/finetune-model-merge.hdf5',verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    tb = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=64,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)
    callbacks_list = [early_stopping, check_point,tb]
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_valid, y_valid), callbacks=callbacks_list)
    # print(history.history)
    # plot_acc(history)
    # plot_loss(history)


if __name__ == '__main__':
    mergeFinetuneModel()

