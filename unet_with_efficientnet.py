# from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Add, LeakyReLU, MaxPooling2D, Dropout, concatenate, Concatenate, Activation
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

from glob import glob 
import numpy as np
import os
import cv2
from PIL import Image


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

# def build_effienet_unet(input_shape):
#     """ Input """
#     inputs = Input(input_shape)

#     """ Pre-trained Encoder """
#     encoder = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

#     s1 = encoder.get_layer("input_1").output                      ## 256
#     s2 = encoder.get_layer("block2a_expand_activation").output    ## 128
#     s3 = encoder.get_layer("block3a_expand_activation").output    ## 64
#     s4 = encoder.get_layer("block4a_expand_activation").output    ## 32

#     """ Bottleneck """
#     b1 = encoder.get_layer("block6a_expand_activation").output    ## 16

#     """ Decoder """
#     d1 = decoder_block(b1, s4, 512)                               ## 32
#     d2 = decoder_block(d1, s3, 256)                               ## 64
#     d3 = decoder_block(d2, s2, 128)                               ## 128
#     d4 = decoder_block(d3, s1, 64)                                ## 256

#     """ Output """
#     outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

#     model = Model(inputs, outputs, name="EfficientNetB0_UNET")
#     return model

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

def create_effienet_unet(input_shape=(None, None, 3),dropout_rate=0.1):
    
    backbone = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)
    
     # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(dropout_rate)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(dropout_rate/2)(uconv0)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
    model = Model(input, output_layer)
    model.name = 'u-xception'

    return model

# def create_data_list(files, input_shape=256):
    
#     images = [] 
#     labels = [] 
    
#     for x in x_train_list:
        
#         # print(x)
        
#         img = cv2.imread(x)
#         # img = cv2.imread(x, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (input_shape, input_shape))
#         # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#         img = img / 255.
#         # img = np.float32(img)
    
#         y = x[:-4] + '.png'
#         # print(y)

#         # if lbl is None:
#         if os.path.isfile(y) == False:
#             print(f'Not found mask file : {x}') 
#             none_mask += 1
#             continue
    
#         lbl = cv2.imread(y, 0)
#         lbl = cv2.resize(lbl, (input_shape, input_shape))
#         lbl[lbl > 0] = 1.
#         lbl = np.float32(lbl)
    
#         images.append(img) 
#         labels.append(lbl) 
#     # 
#     images = np.reshape(images, [-1, input_shape, input_shape, 3])
#     labels = np.reshape(labels, [-1, input_shape, input_shape, 1])
    
#     return images, labels

# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, train_im_path=train_im_path,train_mask_path=train_mask_path,
#                  augmentations=None, batch_size=batch_size,img_size=256, n_channels=3, shuffle=True):
#         'Initialization'
#         self.batch_size = batch_size
#         self.train_im_paths = glob.glob(train_im_path+'/*')
        
#         self.train_im_path = train_im_path
#         self.train_mask_path = train_mask_path

#         self.img_size = img_size
        
#         self.n_channels = n_channels
#         self.shuffle = shuffle
#         self.augment = augmentations
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.ceil(len(self.train_im_paths) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.train_im_paths))]

#         # Find list of IDs
#         list_IDs_im = [self.train_im_paths[k] for k in indexes]

#         # Generate data
#         X, y = self.data_generation(list_IDs_im)

#         if self.augment is None:
#             return X,np.array(y)/255
#         else:            
#             im,mask = [],[]   
#             for x,y in zip(X,y):
#                 augmented = self.augment(image=x, mask=y)
#                 im.append(augmented['image'])
#                 mask.append(augmented['mask'])
#             return np.array(im),np.array(mask)/255

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.train_im_paths))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def data_generation(self, list_IDs_im):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((len(list_IDs_im),self.img_size,self.img_size, self.n_channels))
#         y = np.empty((len(list_IDs_im),self.img_size,self.img_size, 1))

#         # Generate data
#         for i, im_path in enumerate(list_IDs_im):
            
#             im = np.array(Image.open(im_path))
#             mask_path = im_path.replace(self.train_im_path,self.train_mask_path)
            
#             mask = np.array(Image.open(mask_path))
            
            
#             if len(im.shape)==2:
#                 im = np.repeat(im[...,None],3,2)

# #             # Resize sample
#             X[i,] = cv2.resize(im,(self.img_size,self.img_size))

#             # Store class
#             y[i,] = cv2.resize(mask,(self.img_size,self.img_size))[..., np.newaxis]
#             y[y>0] = 255

#         return np.uint8(X),np.uint8(y)
    
def create_data_list(list_IDs_im, img_size):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((len(list_IDs_im), img_size, img_size, 3))
    y = np.empty((len(list_IDs_im), img_size, img_size, 1))

    # Generate data
    for i, im_path in enumerate(list_IDs_im):
        
        im = np.array(Image.open(im_path))
        
        mask_path = im_path[:-4] + '.png'

        if os.path.isfile(mask_path) == False:
            print(f'Not found mask file : {mask_path}') 
            continue
        
        # mask_path = im_path.replace( train_im_path, train_mask_path)
        
        mask = np.array(Image.open(mask_path))
        
        
        if len(im.shape)==2:
            im = np.repeat(im[...,None],3,2)

#             # Resize sample
        X[i,] = cv2.resize(im,( img_size, img_size))

        # Store class
        y[i,] = cv2.resize(mask,( img_size, img_size))[..., np.newaxis]
        y[y>0] = 255

    return np.uint8(X),np.uint8(y)

    
if __name__ == "__main__":
    input_shape = (256, 256, 3)
    input_res = input_shape[0]
    
    # base_path = 'C:/Users/user/Desktop/datasets/Atopy Segmentation'
    base_path = '/Users/sehwanyoo/Dropbox/WORK/SNUH/Atopy Segmentation'

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
    
    train_images, train_labels = create_data_list(x_train_list, input_res)
    
    # print(train_images)
    
    # extra_images, extra_labels = create_data_list(x_extra_list, input_res)
    
    print(f'train_images : {train_images.shape}')
    print(f'train_labels : {train_labels.shape}')
    
    # model = build_effienet_unet(input_shape, )
    model = create_effienet_unet(input_shape=(input_res, input_res, 3))
    model.summary()
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(16, drop_remainder=True)
    
    model.compile(loss='mse',
                  optimizer='adam'
                    # metrics=metrics,
    )
    
    model.fit(train_ds, epochs=10)
