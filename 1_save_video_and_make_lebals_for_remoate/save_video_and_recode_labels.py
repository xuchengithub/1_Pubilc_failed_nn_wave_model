# organize imports
import numpy as np
import cv2
import pickle
import time
# This will return video from the first webcam on your computer.

now_time = time.asctime(time.localtime(time.time()))
now_time = now_time.replace(" ", "_")
now_time = now_time.replace(":", "-")
now_time_avi_fold = "./data/" + now_time + ".avi"
now_time_txt_fold = "./data/" + now_time + ".txt"

cap = cv2.VideoCapture(
    
)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(
    now_time_avi_fold, fourcc, 25, (640, 480)
)  # Make sure (width,height) is the shape of input frame from video
font = cv2.FONT_HERSHEY_SIMPLEX
train_data_list = list()
while True:
    ret, frame = cap.read()
    if ret:
        if cv2.waitKey(1) == ord("1"):
            train_data_list.append(1)
            wave_or_not = 1
        else:
            train_data_list.append(0)
            wave_or_not = 0
        print(len(train_data_list))
        out.write(frame)
        wave_or_not = str(wave_or_not)
        frame = cv2.putText(frame, wave_or_not, (0, 20),  # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
                            font, 0.8, (255, 255, 255), 1)  # The original input frame is shown in the window
        cv2.imshow("Original", frame)
    # Wait for 'a' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord("a"):
        with open(now_time_txt_fold, "wb") as fp:
            pickle.dump(train_data_list, fp)
        break

# Close the window / Release webcam
cap.release()
# After we release our webcam, we also release the output
out.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
