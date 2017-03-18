
# misc helpers
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#####################
# Data parsing      #
#####################

output = "output/"
output_augmented = None

drive_logs_center = open("track1_stay_center/driving_log.csv").readlines()

drive_logs_curves = open("track1_curves/driving_log.csv").readlines()
drive_logs_recover = open("track1_recover/driving_log.csv").readlines()
#drive_logs_track2 = open("track2/driving_log.csv").readlines()
#drive_logs_track2_new = open("track2_new/driving_log.csv").readlines()

#drive_logs = drive_logs_center + drive_logs_curves + drive_logs_recover + drive_logs_track2 + drive_logs_track2_new

drive_logs = drive_logs_center + drive_logs_curves + drive_logs_recover

drive_logs_prepared = open("driving_prepared_data.csv").readlines()

def get_data_set(dataset):
        path = output
        if output_augmented:
                path = output + output_augmented
        nb_lines        = len(dataset)
        print("Data size",nb_lines)
        batch_size      = 32
        correction      = 0.1
        images          = []
        angles          = []
        printed         = False
        for offset in range(0, nb_lines, batch_size):
                batch_lines = dataset[offset:offset+batch_size]

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
                        image_left_flip   = np.fliplr(image_right)
                        image_right_flip  = np.fliplr(image_left)

                        label_left        = label + correction
                        label_right       = label - correction
                        label_left_flip   = -label_right
                        label_right_flip  = -label_left

                        images.append(image)
                        angles.append(label)
                        if output_augmented:
                                images.append(image_left)
                                angles.append(label_left)
                                images.append(image_right)
                                angles.append(label_right)

                                # flipped images 
                                images.append(image_center_flip)
                                angles.append(-label)
                                images.append(image_left_flip)
                                angles.append(label_left_flip)
                                images.append(image_right_flip)
                                angles.append(label_right_flip)

                        if not printed and label != 0.0 and output_augmented:
                            fig_orig, ax_orig = plt.subplots(1, 3)
                            fig_orig.suptitle("Left center and right camera angles", fontsize=12, horizontalalignment="center", verticalalignment="bottom")

                            ax_orig[0].imshow(image_left)
                            ax_orig[0].set_title(str(label_left), fontsize=8, loc="left")
                            ax_orig[0].axis("off")

                            ax_orig[1].imshow(image)
                            ax_orig[1].set_title(str(label), fontsize=8, loc="left")
                            ax_orig[1].axis("off")

                            ax_orig[2].imshow(image_right)
                            ax_orig[2].set_title(str(label_right), fontsize=8, loc="left")
                            ax_orig[2].axis("off")
                            
                            plt.subplots_adjust(top=1.5)
                            plt.savefig(path + "augmented_data_orig.jpg", bbox_inches='tight', pad_inches = 0)
                            plt.close()

                            fig_flipped, ax_flipped = plt.subplots(1, 3)
                            fig_flipped.suptitle("Left center and right camera angles flipped", fontsize=12)

                            ax_flipped[0].imshow(image_left_flip)
                            ax_flipped[0].set_title(str(label_left_flip), fontsize=8, loc="left")
                            ax_flipped[0].axis("off")

                            ax_flipped[1].imshow(image_center_flip)
                            ax_flipped[1].set_title(str(-label), fontsize=8, loc="left")
                            ax_flipped[1].axis("off")

                            ax_flipped[2].imshow(image_right_flip)
                            ax_flipped[2].set_title(str(label_right_flip), fontsize=8, loc="left")
                            ax_flipped[2].axis("off")

                            plt.subplots_adjust(top=1.5)
                            plt.savefig(path + "augmented_data_flipped.jpg", bbox_inches='tight', pad_inches = 0)
                            plt.close()
                            printed = True

        return images, angles

def create_plot(data_set, name, normed=False):
        n, ret_bins, patches = plt.hist(data_set, bins="auto", normed=normed)
        print("Max n",np.max(n))
        print("N;",n)
        print("Bins",ret_bins)
#        print("Maximal bin",ret_bins[n == np.max(n)])
        print("Maximal first 6 bins",n[np.argpartition(n,-6)][-6:])
        maxindices = np.argmax(n)
        print(maxindices)
        print("Max test", n[maxindices])
        if normed:
                plt.ylabel('probability density')
        else:
                plt.ylabel('count')
        plt.xlabel('steering angles')
        plt.title('Histogram of '+name+' driving angles')
        path = output
        if output_augmented:
                path = output + output_augmented
        plt.savefig(path + "result_hist_"+name+".jpg")
        plt.close()

def analyse():
        images, angles = get_data_set(drive_logs_curves)
        print("Angles",len(angles))
        create_plot(angles, "curves", True)
        create_plot(angles, "curves_count")

        images, angles = get_data_set(drive_logs_center)
        print("Angles",len(angles))
        create_plot(angles, "center", True)
        create_plot(angles, "center_count")

        images, angles = get_data_set(drive_logs_recover)
        print("Angles",len(angles))
        create_plot(angles, "recover", True)
        create_plot(angles, "recover_count")

        images, angles = get_data_set(drive_logs)
        print("Angles",len(angles))
        create_plot(angles, "all", True)
        create_plot(angles, "all_count")

        images, angles = get_data_set(drive_logs_prepared)
        print("Angles",len(angles))
        create_plot(angles, "all_prepared", True)
        create_plot(angles, "all_prepared_count")
        return images, angles

angs = []
imgs = []
imgs, angs = analyse()

fig, axarr = plt.subplots(4, 5)
fig.suptitle("Some sample images from recorded data", fontsize=16)

i = 0
j = 0
prev_angles = []
for idx, angle in enumerate(angs):
        if angle in prev_angles:
                continue
        axarr[i, j].imshow(imgs[idx])
        axarr[i, j].set_title(str(angle), fontsize=8, loc="left")
        axarr[i, j].axis("off")
        if i == 3 and j == 4:
                break
        if j == 4:
                i += 1
                j = 0
        else:
                j += 1
        prev_angles.append(angle)

plt.savefig(output + "sample_images.jpg")
plt.close()

output_augmented = "augmented/"
analyse()