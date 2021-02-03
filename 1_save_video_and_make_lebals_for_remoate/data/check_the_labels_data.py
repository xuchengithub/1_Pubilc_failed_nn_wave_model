import pickle
import sys
import os
import numpy as np

fold_address = "./Thu_Jan_21_13-30-36_2021.txt"
# fold_rebuild_csv = f"/home/sctmaintainer/Desktop/open_pose/nursecalldocker/openpose-docker/docker-inside/train_data/data_used_for_train_model_all_{now_time}.npy"
print(os.path)
if os.path.exists(fold_address):
    with open(fold_address, "rb") as fp:
        data = pickle.load(fp)
        data = np.array(data)

        print(type(data))
        print(data.shape)

        print(data)
        label_1 = np.where(data == 1)
        print(label_1)

else:
    print("no have txt fold")
    sys.exit(1)
