import pickle
import os
import sys
import numpy as np


def my_mean(data):
    return sum(data, 0) / sum(data != 0, 0)


def my_std(data, mean):
    return np.sqrt(sum((data - mean)**2, 0) / sum(data != 0, 0))


def export_training_data(data, labels, output_path='data_used_for_train_model_all.npy'):
    if os.path.exists(output_path):
        print('File does exist already!')
        print('If you want to run this script anyway delete or rename the old file first!')
        sys.exit(1)

    fps_video = 25
    video_second_for_pose_recognition = 3
    len_for_time = video_second_for_pose_recognition * fps_video
    num_of_pose_point = data.shape[1]
    data_used_for_train_model_all = np.zeros(
        (data.shape[0] - len_for_time, len_for_time, num_of_pose_point + 1),
        dtype=np.float32
    )

    label_data = list()
    for i in range(len(data)):
        if len_for_time <= i:
            for row_data in range(len_for_time):
                data_used_for_train_model_all[i-len_for_time][row_data][:-1] = \
                    data[i - len_for_time + row_data, :]
            data_used_for_train_model_all[i-len_for_time][row_data][-1] = labels[i]

    print("Saving numpy data...")
    np.save(
        output_path,
        data_used_for_train_model_all,
    )
    print("OK")

    # data_used_for_train_model_all = np.zeros(
    #     (data.shape[0] - len_for_time, len_for_time, num_of_pose_point + 1),
    #     dtype=np.float32
    # )
    #
    # label_data = list()
    # for i in range(len(data)):
    #     if len_for_time <= i:
    #         for row_data in range(len_for_time):
    #             data_used_for_train_model_all[i-len_for_time][row_data][:-1] = \
    #                 data[i - len_for_time + row_data]
    #         data_used_for_train_model_all[i-len_for_time][row_data][-1] = labels[i]
    # np.save(
    #     output_path,
    #     data_used_for_train_model_all,
    # )


def load_data(file):
    if os.path.exists(file):
        with open(file, "rb") as fp:
            data = pickle.load(fp)
        return np.array(data)
    else:
        print(f"No file found at {fold_address}")
        sys.exit(1)


def check_argin(args):
    if (len(args) > 1):
        fold_address = args[1]
        print(f"Using {fold_address} as input")
        output = fold_address[:-4] + '.npy'
        print(f"Saving output to {output}")
    else:
        print("Need path to recording data.")
        print("> python3 preprocess_data /path/file.txt")
        sys.exit(1)
    return fold_address, output


if __name__ == "__main__":
    fold_address, output = check_argin(sys.argv)
    data = load_data(fold_address)

    # extract labels
    labels = data[:,0,-1]
    # remove label from structure
    data = data[:,0,:-1]

    # extract x and y
    xs = data[:, 0::3]
    ys = data[:, 1::3]

    # calculate mean
    mxs = my_mean(xs)
    mys = my_mean(ys)

    # calculate std
    stdxs = my_std(xs, mxs)
    stdys = my_std(ys, mys)

    norm_x = (xs - mxs) / stdxs
    norm_y = (ys - mys) / stdys

    # combine x and y to matrix with [x1, y1, x2, y2, ...]
    training_output = np.zeros((norm_x.shape[0], 2*norm_x.shape[1]))
    training_output[:, 0::2] = norm_x
    training_output[:, 1::2] = norm_y

    export_training_data(training_output, labels, output_path=output)
