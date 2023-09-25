import numpy as np
import cv2
import math
import sys 
import os 
from PIL import Image, ImageDraw, ImageFont


# 把视频缩小到指定尺寸 并旋转 

INPUT_VIDEO_PATH  = '/home/hehanlong/wechat/blibli_mic3_720x1280.mp4'
OUTPUT_VIDEO_PATH  = '/home/hehanlong/wechat/blibli_mic3_180x320.mp4'
TARGET_PNG_DIR    = '/home/hehanlong/wechat/blibli_mic3_180x320/'
IS_ROTATE         = False 
TARGET_WIDTH      = 180
TARGET_HEIGHT     = 320




# 创建目标目录
os.makedirs(TARGET_PNG_DIR, exist_ok=True)

# 打开源视频 
input_video = cv2.VideoCapture(INPUT_VIDEO_PATH)

# 输出视频 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (TARGET_WIDTH, TARGET_HEIGHT)) 

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
    if IS_ROTATE: # 如果旋转 说明原来是横屏 
        width  = TARGET_HEIGHT
        height = TARGET_WIDTH 
 
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)

    if frame_num_a == 1: # 只打印一次
        print(f"resized to {dim}")
 

    resized_bgra = cv2.cvtColor(resized, cv2.COLOR_RGB2BGRA)
    if IS_ROTATE:
        resized_bgra = cv2.rotate(resized_bgra, cv2.ROTATE_90_CLOCKWISE)
        if frame_num_a == 1: 
            print(f"rotate to {resized.shape}  {resized.dtype}")
  

    if frame_num_a >= 810 and frame_num_a <= 1810:
        output = f"{TARGET_PNG_DIR}image_{frame_num_a - 810}.rgba"
        print(f"output = {output}")
        output_video.write(resized_bgra) 
        resized_bgra.tofile(output)
        print(f"resized_bgra {resized_bgra.shape}")
        

    frame_num_a = frame_num_a + 1 


print(f"frame_num_a = {frame_num_a}")
input_video.release()
output_video.release()