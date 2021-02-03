# get the video and labels
#-------------------------------
# use  /home/xuchen/Desktop/docker-inside/1_save_video_and_make_lebals_for_remoate/set_up.sh
# recode the video and labels
# upload to server

# move the video to server get features
#-------------------------------

# save the features and labels to 2_from_the_server_use_openpose_to_get_train_data
#-------------------------------

# use the use_txt_make_numpy.py to merge all the data
#-------------------------------

# ***preprocess the data***
#-------------------------------

# use Train_model.py
#-------------------------------

# check tensorboard
#-------------------------------

# use the model
# -------------------------------
python3 16_tensorboard.py
tensorboard --logdir=runs