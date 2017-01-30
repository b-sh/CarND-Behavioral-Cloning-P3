import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

images=[]
labels=[]

# this is pretty memory consuming

drive_log= open("data/training/driving_log.csv").readlines()

i=0
for line in drive_log:
    line_split=line.split(",")
    img=line_split[0]
    label=line_split[3]

#    if i == 20:
    image = mpimg.imread(img)
    images.append(image)
    labels.append(label)
#        i = 0
#    i += 1

def other_data():
    drive_log= open("track1/driving_log.csv").readlines()

    for line in drive_log:
        line_split=line.split(",")
        label=line_split[3]

        img=line_split[0]
        image = mpimg.imread(img)
        images.append(image)
        labels.append(label)
        img=line_split[1]
        image = mpimg.imread(img.split(" ")[1])
        images.append(image)
        labels.append(label)
        img=line_split[2]
        image = mpimg.imread(img.split(" ")[1])
        images.append(image)
        labels.append(label)

np.savez('data/features.npz', images)
np.savez('data/labels.npz', labels)
