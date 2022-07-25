from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

from glob import glob 
import numpy as np
import os
import cv2
print("TF Version: ", tf.__version__)

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_effienet_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained Encoder """
    encoder = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = encoder.get_layer("input_1").output                      ## 256
    s2 = encoder.get_layer("block2a_expand_activation").output    ## 128
    s3 = encoder.get_layer("block3a_expand_activation").output    ## 64
    s4 = encoder.get_layer("block4a_expand_activation").output    ## 32

    """ Bottleneck """
    b1 = encoder.get_layer("block6a_expand_activation").output    ## 16

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                               ## 32
    d2 = decoder_block(d1, s3, 256)                               ## 64
    d3 = decoder_block(d2, s2, 128)                               ## 128
    d4 = decoder_block(d3, s1, 64)                                ## 256

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="EfficientNetB0_UNET")
    return model

def train_list(files, input_shape=256):
    
    train_images = [] 
    train_labels = [] 
    
    for x in x_train_list:
        img = cv2.imread(x)
        # img = cv2.imread(x, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_shape, input_shape))
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img / 255.
        # img = np.float32(img)
    
        y = x[:-4] + '.png'
        # print(y)

        # if lbl is None:
        if os.path.isfile(y) == False:
            print(f'Not found mask file : {x}') 
            none_mask += 1
            continue
    
        lbl = cv2.imread(y, 0)
        lbl = cv2.resize(lbl, (input_shape, input_shape))
        lbl[lbl > 0] = 1.
        lbl = np.float32(lbl)
    
        train_images.append(img) 
        train_labels.append(lbl) 
    # 
    train_images = np.reshape(train_images, [-1, input_shape, input_shape, 3])
    train_labels = np.reshape(train_labels, [-1, input_shape, input_shape, 1])
    
    return train_images, train_labels
    
if __name__ == "__main__":
    input_shape = (256, 256, 3)
    input_res = input_shape[0]
    
    base_path = 'C:/Users/user/Desktop/datasets/Atopy Segmentation'

    paths = ['Intersect_0.75', 'Intersect_0.8', 'Intersect_0.85']
    grades = ['Grade0', 'Grade1', 'Grade2', 'Grade3']

    path = paths[2]
    grade = grades[0]
    
    x_train = {}
    y_train = {}

    x_test = {}
    y_test = {}

    x_extra = {}
    y_extra = {} 


    # for path in paths:  
    # for grade in grades:
    x_train_list = glob(os.path.join(base_path, path, 'Atopy_Segment_Train', f'{grade}/*.jpg'))
    y_train_list = glob(os.path.join(base_path, path, 'Atopy_Segment_Train', f'{grade}/*.png'))

    x_test_list = glob(os.path.join(base_path, path, 'Atopy_Segment_Test', f'{grade}/*.jpg'))
    y_test_list = glob(os.path.join(base_path, path, 'Atopy_Segment_Test', f'{grade}/*.png'))

    x_extra_list = glob(os.path.join(base_path, path, 'Atopy_Segment_Extra', f'{grade}/*.jpg'))
    y_extra_list = glob(os.path.join(base_path, path, 'Atopy_Segment_Extra', f'{grade}/*.png'))
    
    train_images, train_labels = train_list(x_train_list, input_res)
    
    model = build_effienet_unet(input_shape, )
    model.summary()
    
    model.compile(loss='mse',
                  optimizer='adam'
                    # metrics=metrics,
    )
    
    model.fit(train_images, train_labels, epochs=10)
