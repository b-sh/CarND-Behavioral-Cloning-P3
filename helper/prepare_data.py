import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

new_log = open("driving_prepared_data.csv","w")

drive_logs_center = open("track1_stay_center/driving_log.csv").readlines()
drive_logs_curves = open("track1_curves/driving_log.csv").readlines()
drive_logs_recover = open("track1_recover/driving_log.csv").readlines()
#drive_logs_track2 = open("track2/driving_log.csv").readlines()
#drive_logs_track2_new = open("track2_new/driving_log.csv").readlines()

#drive_logs = drive_logs_center + drive_logs_curves + drive_logs_recover + drive_logs_track2 + drive_logs_track2_new
drive_logs = drive_logs_center + drive_logs_curves + drive_logs_recover

i = 0
for line in drive_logs:
    line_split  =   line.split(",")
    label       =   float(line_split[3])

    if label != 0.0:
        new_log.write(line)
        continue

    if (i == 0 and label == 0.0):
        new_log.write(line)

    # take each xth of 0.0 occurence
    if i == 15:
        i = 0
        continue

    i += 1

new_log.close()