import numpy as np
import cv2
import math
import sys 
import os 
from PIL import Image, ImageDraw, ImageFont


# 把视频缩小到指定尺寸 并旋转 
INPUT_VIDEO_PATH  = '/Users/hehanlong/Downloads/ThirdVideo/Office_Road.mp4'
TARGET_WIDTH      = 720
TARGET_HEIGHT     = 1280

# 自动生成路径 
TARGET_PNG_DIR     = f'{INPUT_VIDEO_PATH[:-4]}_{TARGET_WIDTH}x{TARGET_HEIGHT}'
 

print(f"TARGET_PNG_DIR    = {TARGET_PNG_DIR}"   )
 
# 创建目标目录
os.makedirs(TARGET_PNG_DIR,   exist_ok=True)
 

# 打开源视频 
input_video = cv2.VideoCapture(INPUT_VIDEO_PATH)

# 获取视频帧的宽度和高度
frame_width  = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num    = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video frame width:{frame_width} height:{frame_height} frame_num:{frame_num}")

frame_num_a = 0 

while input_video.isOpened():
    ret, img = input_video.read()
    if not ret:
        print(f"end of video read")
        break

    width  = TARGET_WIDTH  
    height = TARGET_HEIGHT 

    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)

    
    print(f"resized to {dim}")  if frame_num_a == 1  else None 
 
    cv2.imwrite(f"{TARGET_PNG_DIR}/{frame_num_a}.png", resized)
   
    frame_num_a = frame_num_a + 1 

print(f"frame_num_a = {frame_num_a}")
input_video.release()
