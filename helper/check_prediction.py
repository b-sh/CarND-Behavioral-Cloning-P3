from keras.models import Model
from keras.layers.normalization import BatchNormalization 
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.image as mpimg
from keras.utils.visualize_util import plot

import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from keras.models import load_model

model =None 

images = []
labels = []

def check_drive_images():
    drive_log= open("track1_beta_long/driving_log.csv","r")

    diff=[]
    for i in range(1000):
        line = drive_log.readline()

        line_split=line.split(",")
        img        = 'track1_beta_long/IMG/'+line_split[0].split('/')[-1]
        label=line_split[3]

        if float(label) != 0:
            print(line)
            print("Original ", label)
            image = mpimg.imread(img)
            image_array = np.asarray(image)
            transformed_image_array = image_array[None, 80:, :, :]
            steering_angle = float(model.predict(transformed_image_array, batch_size=1))
  
            print("Predicted ", steering_angle)
            diff.append(float(label) - steering_angle)

    print("Max error", max(diff))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='playground')
    parser.add_argument('model', type=str,
    help='Path to model definition h5. Model should be on the same path.')
    args = parser.parse_args()

    model = load_model(args.model)

    check_drive_images()