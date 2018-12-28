# coding: utf-8


import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Lambda
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import adam, Adam
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model


# -------------------------------------
# 定义参数
# -------------------------------------
EPOCHS = 10
BATCH_SIZE = 64
IMAGE_SIZE = (480, 480)
INCEPTIONV3_NO_TRAINABLE_LAYERS = 88
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 3
CLASS_NUM = 10
DROP_RATE = 0.5


# -------------------------------------
# 文件路径
# -------------------------------------
TRAIN_DATA_PATH = './data/train/'
VALID_DATA_PATH = './data/valid/'
MODEL_PATH = './model/'
MODEL_NAME = 'inceptionv3-finetune-model.hdf5'

def finetuneModel():
    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, batch_size=BATCH_SIZE)
    inputs = Input((*IMAGE_SIZE, 3))
    x = Lambda(inception_v3.preprocess_input)(inputs)
    base_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D(name='my_global_average_pooling_layer_1')(base_model.output)
    x = Dropout(DROP_RATE, name='my_dropout_layer_1')(x)
    predictions = Dense(CLASS_NUM, activation='softmax', name='my_dense_layer_1')(x)
    model = Model(base_model.input, predictions)
    plot_model(model, to_file='inceptionv3.png', show_shapes=True)

    # set trainable layer
    for layer in model.layers[:INCEPTIONV3_NO_TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[INCEPTIONV3_NO_TRAINABLE_LAYERS:]:
        layer.trainable = True

    # check
    layers = zip(range(len(model.layers)), [x.name for x in model.layers])
    for layer_num, layer_name in layers:
        print('{}: {}'.format(layer_num + 1, layer_name))

    # check point
    check_point = ModelCheckpoint(monitor='val_loss',
                                  filepath='./model/{}'.format(MODEL_NAME),
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto')

    # early stoppiing
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=0, mode='auto')


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


def extractFeatures():
    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=False, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, shuffle=False, batch_size=BATCH_SIZE)
    inputs = Input((*IMAGE_SIZE, 3))
    x = Lambda(inception_v3.preprocess_input)(inputs)
    base_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    # 加载模型参数
    model.load_weights(os.path.join(MODEL_PATH, MODEL_NAME), by_name=True)
    train_features = model.predict_generator(train_generator, steps=len(train_generator.filenames) // BATCH_SIZE, use_multiprocessing=True, workers=8, verbose=1)
    valid_features = model.predict_generator(valid_generator, steps=len(valid_generator.filenames) // BATCH_SIZE, use_multiprocessing=True, workers=8, verbose=1)
    with h5py.File('inceptionv3-finetune-output.hdf5', 'w') as h:
        h.create_dataset('X_train', data=train_features)
        h.create_dataset('y_train', data=train_generator.classes[:((train_generator.samples // BATCH_SIZE) * BATCH_SIZE)])
        h.create_dataset('X_val', data=valid_features)
        h.create_dataset('y_val', data=valid_generator.classes[:((valid_generator.samples // BATCH_SIZE) * BATCH_SIZE)])


if __name__ == '__main__':
    finetuneModel()
    extractFeatures()
