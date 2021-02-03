import numpy as np
from copy import deepcopy
import pickle
import csv
import os
import sys

# --------------------------import data-----------------------------
fold_address_label = "/home/xuchen/Desktop/docker-inside/2_from_the_server_use_openpose_to_get_train_data/data5/Sat_Jan_23_01-17-49_2021.txt"
fold_address = "/home/xuchen/Desktop/docker-inside/2_from_the_server_use_openpose_to_get_train_data/data5/pose_recognition_data.txt"
fold_rebuild_csv = "/home/xuchen/Desktop/docker-inside/2_from_the_server_use_openpose_to_get_train_data/data5/data_used_for_train_model_all.npy"
if os.path.exists(fold_address_label):
    with open(fold_address_label, "rb") as label_fp:
        label_data = pickle.load(label_fp)
        print(type(label_data))
else:
    print("no have txt fold")
    sys.exit(1)

if os.path.exists(fold_address):
    with open(fold_address, "rb") as fp:
        data = pickle.load(fp)
        print(type(data[1]))
        print((data[1].shape))
        print(len(data))
else:
    print("no have txt fold")
    sys.exit(1)


for i in range(len(data)-1):
    data[i][0][201] = label_data[i]
    print(data[i][0][201])


data_used_for_train_model_all = list()
if os.path.exists(fold_rebuild_csv) == False:
    fps_video = 25
    video_second_for_pose_recognition = 3
    len_for_time = video_second_for_pose_recognition * fps_video
    num_of_pose_point = data[1].shape[1] - 1
    sizecol_of_feature = int((num_of_pose_point) / 3 * 2)
    data_used_for_train_model_one = np.zeros(
        (len_for_time, sizecol_of_feature), dtype=np.float32
    )

    label_data = list()
    for i in range(len(data)):
        if len_for_time <= i:

            for row_data in range(len_for_time):
                col = 0
                for column_data in range(sizecol_of_feature):
                    if col % 3 == 2:
                        col = col + 1

                    data_used_for_train_model_one[row_data][column_data] = data[
                        i - len_for_time + row_data
                    ][0][col]

                    col = col + 1

            b = data[i - len_for_time][0][201]
            tuples_data = (
                deepcopy(data_used_for_train_model_one), deepcopy(b))
            data_used_for_train_model_all.append(tuples_data)
    np.save(
        fold_rebuild_csv,
        data_used_for_train_model_all,
    )
