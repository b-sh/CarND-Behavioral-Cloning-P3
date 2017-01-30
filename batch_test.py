from keras.models import Model
from keras.layers.normalization import BatchNormalization 
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.image as mpimg
from keras.utils.visualize_util import plot

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

X_train=np.load("data/features_red.npz")['arr_0']
Y_train=np.load("data/labels_red.npz")['arr_0']

nb_epoch=1

train_datagen = ImageDataGenerator(
#                samplewise_center=False,
#                featurewise_std_normalization=False,
#                samplewise_std_normalization=True,
#                zca_whitening=False,
                rotation_range=10,
                width_shift_range=0.3,
                height_shift_range=0.3,
#                shear_range=0.,
#                zoom_range=0.,
#                channel_shift_range=0.,
#                fill_mode='nearest',
#                cval=0.,
#                horizontal_flip=True,
#                vertical_flip=True,
#                rescale=None,
                rescale=1./255,
#                shear_range=0.2,
                zoom_range=0.2,
#                horizontal_flip=True
                )

for e in range(nb_epoch):
    print('Epoch', e)
    batches = 0
    print("Y train", Y_train)
    for X_batch, Y_batch in train_datagen.flow(X_train, Y_train, batch_size=32, save_to_dir="preview"):
        print("")
        print("Y batch", Y_batch)
        batches += 1
        if batches > 1:
            break;
    break;