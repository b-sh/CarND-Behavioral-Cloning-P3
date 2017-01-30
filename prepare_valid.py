import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

images=[]
labels=[]
drive_log= open("data/validation/driving_log.csv").readlines()

del drive_log[0]

for line in drive_log:
    line_split=line.split(",")
    label=line_split[3]

    img=line_split[0]
    image = mpimg.imread("data/validation/"+img)
    images.append(image)
    labels.append(label)
    img=line_split[1]
    image = mpimg.imread("data/validation/"+img.split(" ")[1])
    images.append(image)
    labels.append(label)
    img=line_split[2]
    image = mpimg.imread("data/validation/"+img.split(" ")[1])
    images.append(image)
    labels.append(label)

plt.imshow(images[0])
plt.show()

np.savez('data/features_valid.npz', images)
np.savez('data/labels_valid.npz', labels)

