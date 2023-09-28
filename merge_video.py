import numpy as np
import cv2
import math
import sys 
import os 
import time
from PIL import Image, ImageDraw, ImageFont


# 合并两个同样大小的视频 

LEFT_VIDEO_PATH   = '/home/hehanlong/wechat/old.mp4'
RIGHT_VIDEO_PATH  = '/home/hehanlong/wechat/new.mp4'
OUTPUT_VIDEO_PATH = '/home/hehanlong/wechat/merge.mp4'
SHOW_ON_FLY   = True # 是否每帧显示 
AUTO_GO_MS   = 1000  # <=0 就会用waitKey 

# 打开源视频 
left_video  = cv2.VideoCapture(LEFT_VIDEO_PATH)
right_video = cv2.VideoCapture(RIGHT_VIDEO_PATH)

# 获取视频帧的宽度和高度
frame_width  = int(left_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(left_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num    = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video frame width:{frame_width} height:{frame_height} frame_num:{frame_num}")


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (frame_width*2, frame_height)) 


frame_num_a = 0 

while left_video.isOpened():
    ret,  img  = left_video.read()
    ret2, img2 = right_video.read() 

    if not ret or not ret2 :
        print(f"end of video read ret={ret} ret2={ret2}")
        break

    merged_img = cv2.hconcat([img, img2])

    if SHOW_ON_FLY:
        cv2.imshow("merged", merged_img)
        if AUTO_GO_MS <= 0:
            cv2.waitKey(5000) 
        else:
            time.sleep(AUTO_GO_MS*1.0/1000.0)

    output_video.write(merged_img) 

    frame_num_a = frame_num_a + 1 


print(f"frame_num_a = {frame_num_a}")
input_video.release()
output_video.release()