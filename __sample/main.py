import tensorflow as tf 
from tensorflow import keras
import os 
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
import cv2

from models import unet 

base_path = 'C:/Users/user/Desktop/datasets/Atopy Segmentation'
paths = ['Intersect_0.75', 'Intersect_0.8', 'Intersect_0.85']
grades = ['Grade0', 'Grade1', 'Grade2', 'Grade3']

num_res = 256
num_classes = [0, 1, 2, 3]
input_shape = (256, 256, 3)

def create_dataset(data_list):
    images = [] 
    masks = [] 
    labels = [] 
    
    for x in data_list:
        img = cv2.imread(x, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (num_res, num_res)) / 255.
        
        y = x[:-4] + '.png'
        
        if os.path.isfile(y) == False:
            print(f'Not found mask file : {x}') 
            # none_mask += 1
            continue
        
        msk = cv2.imread(y, 0)
        msk = cv2.resize(msk, (num_res, num_res))
        msk[msk > 0] = 1.
        msk = np.float32(msk)

        lbl = np.float32(y.split('\\')[-2][-1])
        
        images.append(img) 
        masks.append(msk) 
        labels.append(lbl) 
    # 
    images = np.reshape(images, [-1, num_res, num_res, 3])
    masks = np.reshape(masks, [-1, num_res, num_res, 1])
    labels = np.reshape(labels, [-1, 1])
    
    return images, masks, labels


if __name__ == '__main__':
    train_list = glob(os.path.join(base_path, paths[2], 'Atopy_Segment_Train', f'*/*.jpg'))

    test_list = glob(os.path.join(base_path, paths[2], 'Atopy_Segment_Test', f'*/*.jpg'))

    extra_list = glob(os.path.join(base_path, paths[2], 'Atopy_Segment_Extra', f'*/*.jpg'))
    
    train_images, train_masks, train_labels = create_dataset(train_list)
    test_images, test_masks, test_labels = create_dataset(test_list)
    extra_images, extra_masks, extra_labels = create_dataset(extra_list)
    
    
    model = unet.build_effienet_unet(input_shape)

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', mode='auto', factor=0.2, patience=5, min_lr=0.0001)
    
    model.compile(
    keras.optimizers.Adam(0.001), 
    loss=[keras.losses.MeanSquaredError(),
          keras.losses.SparseCategoricalCrossentropy()],
    metrics=['accuracy'],)
    
    hist = model.fit(train_images,
                     [train_masks, train_labels],
                     batch_size=16,
                     epochs=20, 
                     validation_data=(extra_images, [extra_masks, extra_labels]),shuffle=True,)
    
    
