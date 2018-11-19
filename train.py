# coding: utf-8


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model
from keras.applications import resnet50
from keras.applications.resnet50 import ResNet50
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications import xception
from keras.applications.xception import Xception
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import adam
from keras.models import load_model
from keras.applications import imagenet_utils
import h5py
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPool1D
from keras.layers.core import Flatten


# -------------------------------------
# 定义参数
# -------------------------------------
EPOCHS = 10
BATCH_SIZE = 64
IMAGE_SIZE = (480, 480)
INCEPTIONV3_NO_TRAINABLE_LAYERS = 88
RESNET50_NO_TRAINABLE_LAYERS = 80
XCEPTION_NO_TRAINABLE_LAYERS = 86
LEARNING_RATE = 1e-4
PATIENCE = 3
GPU_NUM = 2
CLASS_NUM = 10
DROP_RATE = 0.5


# -------------------------------------
# 文件路径
# -------------------------------------
TRAIN_DATA_PATH = './data/train/'
VALID_DATA_PATH = './data/valid/'
MODEL_PATH = './model/'


def finetuneModel(preprocess_input_func, base_model_class, model_name, no_trainable_layers,
                        pooling_way='global_average_pooling'):
    """
    preprocess_input_func: 数据预处理函数
    base_model_class: 类
    model_name: 模型名
    no_trainable_layers: 不训练的层数
    pooling_way: 池化方式
    """

    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, batch_size=BATCH_SIZE)

    inputs = Input((*IMAGE_SIZE, 3))
    x = Lambda(preprocess_input_func)(inputs)
    base_model = base_model_class(input_tensor=x, weights='imagenet', include_top=False)
    if pooling_way == 'global_average_pooling':  # 默认为全局平均池化
        x = GlobalAveragePooling2D()(base_model.output)
    else:
        x = GlobalMaxPooling2D()(base_model.output)
    x = Dropout(DROP_RATE)(x)
    predictions = Dense(CLASS_NUM, activation='softmax')(x)
    model = Model(base_model.input, predictions)

    layers = zip(range(len(model.layers)), [x.name for x in model.layers])
    for layer_num, layer_name in layers:
        print('{}: {}'.format(layer_num + 1, layer_name))
    plot_model(model, to_file='{}.png'.format(model_name), show_shapes=True)

    for layer in model.layers[:no_trainable_layers]:
        layer.trainable = False
    for layer in model.layers[no_trainable_layers:]:
        layer.trainable = True
    model = multi_gpu_model(model, GPU_NUM)

    # check point
    check_point = ModelCheckpoint(monitor='val_loss',
                                    filepath='model/{}-finetune.hdf5'.format(model_name),
                                    verbose=1,
                                    save_best_only=True,
                                    mode='auto')

    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='auto')

    # compile
    model.compile(optimizer=adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    # fit
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        callbacks=[check_point, early_stopping]
    )



def extractFeatures(pooling_way='global_average_pooling'):

    filenames = ['xception-finetune.hdf5', 'resnet50-finetune.hdf5', 'inceptionv3-finetune.hdf5']

    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=False, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, shuffle=False, batch_size=BATCH_SIZE)

    for filename in filenames:

        inputs = Input((*IMAGE_SIZE, 3))
        if filename == 'xception-finetune.hdf5':
            x = Lambda(xception.preprocess_input)(inputs)
            base_model = Xception(input_tensor=x, weights='imagenet', include_top=False)
        elif filename == 'resnet50-finetune.hdf5':
            x = Lambda(resnet50.preprocess_input)(inputs)
            base_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
        elif filename == 'inceptionv3-finetune.hdf5':
            x = Lambda(inception_v3.preprocess_input)(inputs)
            base_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
        if pooling_way == 'global_average_pooling':
            model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
        else:
            model = Model(base_model.input, GlobalMaxPooling2D()(base_model.output))

        # 加载模型参数
        model_weights_path = os.path.join(MODEL_PATH, filename)
        model.load_weights(model_weights_path, by_name=True)
        # model.summary()

        train_features = model.predict_generator(train_generator, steps=len(train_generator.filenames) // BATCH_SIZE, use_multiprocessing=True, workers=8, verbose=1)
        valid_features = model.predict_generator(valid_generator, steps=len(valid_generator.filenames) // BATCH_SIZE, use_multiprocessing=True, workers=8, verbose=1)

        with h5py.File('{}-output.hdf5'.format(filename[:-5]), 'w') as h:
            h.create_dataset('X_train', data=train_features)
            h.create_dataset('y_train', data=train_generator.classes[:((train_generator.samples // BATCH_SIZE) * BATCH_SIZE)])
            h.create_dataset('X_val', data=valid_features)
            h.create_dataset('y_val', data=valid_generator.classes[:((valid_generator.samples // BATCH_SIZE) * BATCH_SIZE)])


def mergeFinetuneModel():
    """
    效果不如单模型，这种融合方式并没有让效果得到提升
    """

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

    print('X_train shape:', X_train.shape)
    print('X_valid shape:', X_valid.shape)
    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)

    inputs = Input(X_train.shape[1:])
    x = Dense(1024, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs, predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    check_point = ModelCheckpoint(filepath='./model/weights.best.merge.hdf5',verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='auto')
    callbacks_list = [early_stopping, check_point]
    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_valid, y_valid), callbacks=callbacks_list)

if __name__ == '__main__':
    # finetuneModel(inception_v3.preprocess_input, InceptionV3, 'inceptionv3', INCEPTIONV3_NO_TRAINABLE_LAYERS)
    # finetuneModel(resnet50.preprocess_input, ResNet50, 'resnet50', RESNET50_NO_TRAINABLE_LAYERS)
    # finetuneModel(xception.preprocess_input, Xception, 'xception', XCEPTION_NO_TRAINABLE_LAYERS)
    # extractFeatures()
    mergeFinetuneModel()
