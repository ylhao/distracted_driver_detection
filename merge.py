
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50, inception_v3, xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers.core import Lambda
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import adam, Adam
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model


# -------------------------------------
# 定义参数
# -------------------------------------
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = (480, 480)
INCEPTIONV3_NO_TRAINABLE_LAYERS = 88
RESNET50_NO_TRAINABLE_LAYERS = 80
XCEPTION_NO_TRAINABLE_LAYERS = 86
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
CLASS_NUM = 10
DROP_RATE = 0.5


# -------------------------------------
# 文件路径
# -------------------------------------
TRAIN_DATA_PATH = './data/train/'
VALID_DATA_PATH = './data/valid/'
MODEL_PATH = './model/'


def mergeModel():

    train_gen = ImageDataGenerator()
    valid_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(TRAIN_DATA_PATH, IMAGE_SIZE, shuffle=True, batch_size=BATCH_SIZE)
    valid_generator = valid_gen.flow_from_directory(VALID_DATA_PATH, IMAGE_SIZE, batch_size=BATCH_SIZE)

    inputs = Input((*IMAGE_SIZE, 3))

    # 定义 resnet
    resnet_x = Lambda(resnet50.preprocess_input)(inputs)
    resnet_base_model = ResNet50(input_tensor=resnet_x, weights='imagenet', include_top=False)
    for layer in resnet_base_model.layers[:RESNET50_NO_TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in resnet_base_model.layers[RESNET50_NO_TRAINABLE_LAYERS:]:
        layer.trainable = True
    resnet_x = GlobalAveragePooling2D(name='my_resnet_global_average_pooling_layer_1')(resnet_base_model.output)

    # 定义 inception
    inception_x = Lambda(inception_v3.preprocess_input)(inputs)
    inception_base_model = InceptionV3(input_tensor=inception_x, weights='imagenet', include_top=False)
    for layer in inception_base_model.layers[:INCEPTIONV3_NO_TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in inception_base_model.layers[INCEPTIONV3_NO_TRAINABLE_LAYERS:]:
        layer.trainable = True
    inception_x = GlobalAveragePooling2D(name='my_inception_global_average_pooling_layer_1')(inception_base_model.output)

    # 定义 xception
    xception_x = Lambda(xception.preprocess_input)(inputs)
    xception_base_model = Xception(input_tensor=xception_x, weights='imagenet', include_top=False)
    for layer in xception_base_model.layers[:XCEPTION_NO_TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in xception_base_model.layers[XCEPTION_NO_TRAINABLE_LAYERS:]:
        layer.trainable = True
    xception_x = GlobalAveragePooling2D(name='my_xception_global_average_pooling_layer_1')(xception_base_model.output)

    # 模型融合：向量拼接
    x = Concatenate(axis=1)([resnet_x, inception_x, xception_x])

    # check
    print(resnet_x.shape, inception_x.shape, xception_x.shape)
    print(resnet_x.shape, inception_x.shape)
    print(x.shape)

    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs, predictions)
    plot_model(model, to_file='merge.png', show_shapes=True)

    # check point
    check_point = ModelCheckpoint(monitor='val_loss',
                                  filepath='./model/merge-model.hdf5',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto')

    # 早停
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=0, mode='auto')

    # 当评价指标不在提升时，减少学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=REDUCE_LR_PATIENCE, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    # 编译模型
    model.compile(optimizer=adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        callbacks=[check_point, early_stopping, reduce_lr]
    )


if __name__ == '__main__':
    mergeModel()
