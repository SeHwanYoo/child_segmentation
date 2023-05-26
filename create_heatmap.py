import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
import os
import cv2 

def get_imgs_lbls(N_RES):
    test_images = [] 
    test_labels = [] 
    none_mask = 0 

    for x in x_test_list:
        img = cv2.imread(x)
        # img = cv2.imread(x, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (N_RES, N_RES))
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
        
        # lbl = cv2.imread(y, cv2.COLOR_BGR2GRAY)
        lbl = cv2.imread(y, 0)
        lbl = cv2.resize(lbl, (N_RES, N_RES))
        # lbl = cv2.normalize(lbl, None, 0, 255, cv2.NORM_MINMAX)
        lbl[lbl > 0] = 1.
        lbl = np.float32(lbl)
        
        
        test_images.append(img) 
        test_labels.append(lbl) 
        
    test_images = np.reshape(test_images, [-1, N_RES, N_RES, 3])
    test_labels = np.reshape(test_labels, [-1, N_RES, N_RES, 1])
    
    return test_images, test_labels 


# level of atopy 
grade = '0'

# Path for heatmap images
heatmap_dir = f'C:/Users/user/Desktop/models/child_segmentation/heatmap_{grade}/'
img_size = 256 
inference_num = 100 

# Dataset path
base_path = '' 

# load trained model
model = tf.keras.models.load_model('')

# Load test images and labels
x_test_list = glob(os.path.join(base_path, 'Atopy_Segment_Test', f'{grade}/*.jpg'))
y_test_list = glob(os.path.join(base_path, 'Atopy_Segment_Test', f'{grade}/*.png'))

# Load as images
test_images, test_labels = get_imgs_lbls(img_size)

# Inference
y_samples = np.stack([model(test_images[:10], training=True) for _ in range(inference_num)])

# Save heatmap images
inx = 0
for kk in range(y_samples[0].shape[0]):
    test_pred_imgs = y_samples[:, kk, :, :, :]
    
    heatmap_img = np.zeros((img_size, img_size), np.uint8)
    for jj in range(inference_num):
        pred_img = np.reshape(test_pred_imgs[jj], [img_size, img_size])        
        heatmap_img = np.add(heatmap_img, pred_img)
        
    new_heatmap = 255 - (heatmap_img / inference_num)
    
    plt.imsave(heatmap_dir + 'heatmap_' + str(inx), new_heatmap, cmap='OrRd')
    inx += 1