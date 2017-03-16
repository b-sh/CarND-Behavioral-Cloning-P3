
# misc helpers
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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


def get_data_set(dataset):
        nb_lines        = len(dataset)
        print(nb_lines)
        batch_size      = 32
        correction      = 0.1
        images          = []
        angles          = []
        for offset in range(0, nb_lines, batch_size):
                batch_lines = drive_logs[offset:offset+batch_size]

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

                        angles.append(label)
                        images.append(image_left)
                        angles.append(label_left)
                        images.append(image_right)
                        angles.append(label_right)

                        # flipped images 
                        images.append(image_center_flip)
                        angles.append(-label)
                        images.append(image_left_flip)
                        angles.append(-label_left)
                        images.append(image_right_flip)
                        angles.append(-label_right)
        return images, angles

def create_plot(data_set, name, normed=False):
        plt.hist(angles, normed=normed)
        if normed:
#                plt.hist(angles, 500, normed=normed)
                plt.ylabel('probability density')
        else:
#                plt.hist(angles)
                plt.ylabel('count')
        plt.xlabel('steering angles')
        plt.title('Histogram of '+name+' driving angles')
        plt.savefig("result_hist_"+name+".jpg")
#        plt.show()

images, angles = get_data_set(drive_logs_center)
print(len(angles))
create_plot(angles, "center", True)
create_plot(angles, "center_count")

images, angles = get_data_set(drive_logs_curves)
print(len(angles))
create_plot(angles, "curves", True)
create_plot(angles, "curves_count")

images, angles = get_data_set(drive_logs_recover)
print(len(angles))
create_plot(angles, "recover", True)
create_plot(angles, "recover_count")

images, angles = get_data_set(drive_logs)
print(len(angles))
create_plot(angles, "all", True)
create_plot(angles, "all_count")
