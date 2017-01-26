# keras
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

# misc helpers
import numpy as np
import matplotlib.image as mpimg
import sklearn

##############
# Model      # 
##############

# inspired by http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
# taken tip from udacity class to be around zero mean and std deviation
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(80, 320, 3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

# saving for docu
plot(model, to_file='model.png', show_shapes=True)

#####################
# Data parsing      #
#####################

drive_logs_center = open("track1_stay_center/driving_log.csv").readlines()
drive_logs_curves = open("track1_curves/driving_log.csv").readlines()
drive_logs_recover = open("track1_recover/driving_log.csv").readlines()
#drive_logs_track2 = open("track2/driving_log.csv").readlines()
#drive_logs_track2_new = open("track2_new/driving_log.csv").readlines()

#drive_logs = drive_logs_center + drive_logs_curves + drive_logs_recover + drive_logs_track2 + drive_logs_track2_new
drive_logs = drive_logs_center + drive_logs_curves + drive_logs_recover

#####################
# Generator setup   #
#####################

# https://keras.io/models/sequential/#sequential-model-methods
# parse for generator to save memory as learned from prepare scripts that it consumes a lot of memory
def generate_arrays_from_file(drive_logs, batch_size=32):
    while True:
        nb_lines   = len(drive_logs)
        correction = 0.1
        for offset in range(0, nb_lines, batch_size):
            batch_lines = drive_logs[offset:offset+batch_size]

            images = []
            labels = []
            for line in batch_lines:
                line_split  = line.split(",")
                img         = line_split[0].split('/')[-3]+'/IMG/'+line_split[0].split('/')[-1]
                img_left    = line_split[1].split('/')[-3]+'/IMG/'+line_split[1].split('/')[-1]
                img_right   = line_split[2].split('/')[-3]+'/IMG/'+line_split[2].split('/')[-1]
                label       = float(line_split[3])
                image       = mpimg.imread(img)
                image_left  = mpimg.imread(img_left)
                image_right = mpimg.imread(img_right)

                # augmenting data
                image_center_flip = np.fliplr(image)
                image_left_flip   = np.fliplr(image_left)
                image_right_flip  = np.fliplr(image_right)

                label_left  = label + correction
                label_right = label - correction

                images.append(image)
                labels.append(label)
                images.append(image_left)
                labels.append(label_left)
                images.append(image_right)
                labels.append(label_right)

                # flipped images 
                images.append(image_center_flip)
                labels.append(-label)
                images.append(image_left_flip)
                labels.append(-label_left)
                images.append(image_right_flip)
                labels.append(-label_right)

            # nice tip from udacity class
            # trim image to only see section with road
            X_train = np.array(images)
            X_train = X_train[:,80:,:,:] 
            y_train = np.array(labels)
            yield sklearn.utils.shuffle(X_train, y_train)

########################
# Training set split   #
########################

from sklearn.model_selection import train_test_split
# using 70% for training and 30% for validation
drive_train, drive_validation = train_test_split(drive_logs, test_size=0.3)

##############
# Training   #
##############

def train_model():
        nb_epoch = 5

        # setup generators for training and validation
        train_generator      = generate_arrays_from_file(drive_train)
        validation_generator = generate_arrays_from_file(drive_validation)

        # using adam optimizer and mean squared error metrics
        model.compile(optimizer='adam',
                loss='mse',
                metrics=['mse'])

        # training model with custom generators
        # original set of data is multiplied with 6 because of (left,right and flip)
        model.fit_generator(
                train_generator,
                samples_per_epoch=6*len(drive_train),
                nb_epoch=nb_epoch,
                validation_data=validation_generator,
                nb_val_samples=6*len(drive_validation))
   
        model.save('model.h5')

train_model()
