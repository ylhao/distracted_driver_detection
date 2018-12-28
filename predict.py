#encoding: utf-8


import os
import argparse
import numpy as np
import keras
import h5py
from keras import backend
from keras.preprocessing.image import load_img, img_to_array
import operator
from keras.applications import resnet50, inception_v3, xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers.core import Lambda
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout
from keras.models import Model


# 解析脚本参数
a = argparse.ArgumentParser(description="Predict the class of a given driver image.")
a.add_argument("--image", help="path to image", default='./test_imgs/1.jpg')
args = a.parse_args()


img_path = args.image
if args.image is not None:
    img_path = args.image


# 定义参数
class_labels = ['safe_driving', 'texting_right', 'talking_on_phone_right', 'texting_left', 'talking_on_phone_left',
                'operating_radio', 'drinking', 'reaching_behind', 'doing_hair_makeup', 'talking_to_passanger']
IMAGE_SIZE = (480, 480)
CLASS_NUM = 10
DROP_RATE = 0


# 加载图片
image = load_img(img_path, target_size=IMAGE_SIZE)
image_arr = img_to_array(image)
image_arr = np.expand_dims(image_arr, axis=0)


# 定义模型
inputs = Input((*IMAGE_SIZE, 3))
x = Lambda(resnet50.preprocess_input)(inputs)
base_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
x = GlobalAveragePooling2D(name='my_global_average_pooling_layer_1')(base_model.output)
x = Dropout(DROP_RATE, name='my_dropout_layer_1')(x)
predictions = Dense(CLASS_NUM, activation='softmax', name='my_dense_layer_1')(x)
model = Model(base_model.input, predictions)


# 加载模型参数
model.load_weights('./model/resnet50-finetune-model.hdf5', by_name=True)


# 预测
predicted = model.predict(image_arr)
decoded_predictions = dict(zip(class_labels, predicted[0]))
decoded_predictions = sorted(decoded_predictions.items(), key=operator.itemgetter(1), reverse=True)
print()
count = 1
for key, value in decoded_predictions[:10]:
    print("{}. {}: {:.4f}%".format(count, key, value*100))
    count+=1
print()

