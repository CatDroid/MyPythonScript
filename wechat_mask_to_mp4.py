import numpy as np
import cv2
import math
import sys 
import os 
from PIL import Image, ImageDraw, ImageFont

# 打开输入源视频  
NAME = "VID_20230927_112805"
INPUT_VIDEO_PATH  = f'/Volumes/MySanDisk/data/Val_2/{NAME}.mp4'
MASK_DIR = f'/Volumes/MySanDisk/data/Val_2/{NAME}_180x320_rgba_output'


parent_path= os.path.dirname(INPUT_VIDEO_PATH)
base_name  = os.path.basename(INPUT_VIDEO_PATH)
short_name = os.path.splitext(base_name)[0]
short_name = short_name + "_mask"

OUTPUT_VIDEO_PATH = os.path.join(parent_path, short_name + '.mp4')   
input_video = cv2.VideoCapture(INPUT_VIDEO_PATH)

# 获取视频帧的宽度和高度
frame_width  = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num    = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video frame width:{frame_width} height:{frame_height} frame_num:{frame_num} output:{OUTPUT_VIDEO_PATH}")


# 定义视频编解码器和输出参数  两倍宽度
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (frame_width, frame_height)) 

frame_num_a = 0 

while input_video.isOpened():
    ret, origin_image = input_video.read()
    if not ret:
        print(f"end of video read")
        break

    mask_file = os.path.join(MASK_DIR, f'SetImage_{frame_num_a}.rgba')
    print(f"open {mask_file}" )
    mask_image = np.fromfile(mask_file, dtype='uint8')
    mask_image = mask_image .reshape(224, 128) # (rows, cols, channels)
    mask_image = np.stack( (mask_image, ) * 3 , axis=-1 ) # 重复3个(224, 126) 合并成 (224, 126, 3)

 
    # 放大到原图
    mask_image_resized = cv2.resize( mask_image,
                    dsize=(origin_image.shape[1], origin_image.shape[0]), 
                    interpolation = cv2.INTER_LINEAR)

    # 写入视频文件 
    output_video.write(mask_image_resized.astype(np.uint8)) 

    # 下一帧
    frame_num_a = frame_num_a + 1 
