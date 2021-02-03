import numpy as np
from copy import deepcopy
import pickle
import os
import sys
import time
import glob
# --------------------------import data-----------------------------
now_time = time.asctime(time.localtime(time.time()))
now_time = now_time.replace(" ", "_")
now_time = now_time.replace(":", "-")
fold_address = "/home/sctmaintainer/Desktop/open_pose/nursecalldocker/openpose-docker/docker-inside/train_data/pose_recognition_data.txt"
path = "/home/xuchen/Desktop/docker-inside/2_from_the_server_use_openpose_to_get_train_data"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
xy = np.zeros((1, 2))
i = 0
for file in files:  # 遍历文件夹
    if os.path.isdir(os.path.join(path, file)):  # 判断是否是文件夹，不是文件夹才打开
        fold_data = os.path.join(path, file)
        os.chdir(fold_data)
        for file in glob.glob("*.npy"):
            txt_fold_address = os.path.join(fold_data, file)
            with open(txt_fold_address, "rb") as fp:
                i = i+1
                if i == 1:
                    first_numpy = np.load(fp, allow_pickle=True)
                    data = first_numpy
                else:
                    next_numpy = np.load(fp, allow_pickle=True)
                    data = np.concatenate((next_numpy, data), axis=0)

# data[frame][feature_or_label][3_second_frame][point]
# --------------------------rebuild data-----------------------------
fold_rebuild_np = "/home/xuchen/Desktop/docker-inside/2_from_the_server_use_openpose_to_get_train_data/all_train_data.npy"
np.save(
    fold_rebuild_np,
    data,
)
