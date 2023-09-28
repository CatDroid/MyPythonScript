import numpy as np
import cv2
import math
import sys 
import os 
from PIL import Image, ImageDraw, ImageFont


# 把视频缩小到指定尺寸 并旋转 

INPUT_VIDEO_PATH  = '/Users/hehanlong/Downloads/NewFlashVideo/VID_20230927_112805.mp4'
IS_ROTATE         = False 
TARGET_WIDTH      = 180
TARGET_HEIGHT     = 320

# 自动生成路径 
OUTPUT_VIDEO_PATH  = f'{INPUT_VIDEO_PATH[:-4]}_{TARGET_WIDTH}x{TARGET_HEIGHT}.mp4'
TARGET_PNG_DIR     = f'{INPUT_VIDEO_PATH[:-4]}_{TARGET_WIDTH}x{TARGET_HEIGHT}'
TARGET_RGBA_DIR    = TARGET_PNG_DIR[:-1] + "_rgba" if TARGET_PNG_DIR[-1]=="/" else TARGET_PNG_DIR + "_rgba"


print(f"OUTPUT_VIDEO_PATH = {OUTPUT_VIDEO_PATH}")
print(f"TARGET_PNG_DIR    = {TARGET_PNG_DIR}"   )
print(f"TARGET_RGBA_DIR   = {TARGET_RGBA_DIR}"  )


# 创建目标目录
#os.makedirs(TARGET_PNG_DIR,   exist_ok=True)
os.makedirs(TARGET_RGBA_DIR,  exist_ok=True)

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

    
    print(f"resized to {dim}")  if frame_num_a == 1  else None 
 

   
    resized_bgra = cv2.cvtColor(resized, cv2.COLOR_RGB2BGRA)
    if IS_ROTATE:
        resized_bgra = cv2.rotate(resized_bgra, cv2.ROTATE_90_CLOCKWISE)
        if frame_num_a == 1: 
            print(f"rotate to {resized.shape}  {resized.dtype}")
  

    output = f"{TARGET_RGBA_DIR}/image_{frame_num_a}.rgba"
    print(f"output = {output}") if frame_num_a == 1 else None 
    output_video.write( cv2.cvtColor(resized_bgra, cv2.COLOR_BGRA2RGB) )  # opencv不支持rgba mp4
    resized_bgra.tofile(output)
    print(f"resized_bgra {resized_bgra.shape}") if frame_num_a == 1 else None  
        

    frame_num_a = frame_num_a + 1 


print(f"frame_num_a = {frame_num_a}")
input_video.release()
output_video.release()