import numpy as np
import cv2
import math
import sys 
import os 
from PIL import Image, ImageDraw, ImageFont

# 打开输入源视频  
INPUT_VIDEO_PATH  = '/Users/hehanlong/Downloads/NewFlashVideo/VID_20230927_111041.mp4'
MASK_DIR = '/Users/hehanlong/Downloads/NewFlashVideo/VID_20230927_111041_180x320_rgba_output'


parent_path= os.path.dirname(INPUT_VIDEO_PATH)
base_name  = os.path.basename(INPUT_VIDEO_PATH)
short_name = os.path.splitext(base_name)[0]
short_name = short_name + "_merged"

OUTPUT_VIDEO_PATH = os.path.join(parent_path, short_name + '.mp4')   
input_video = cv2.VideoCapture(INPUT_VIDEO_PATH)

# 获取视频帧的宽度和高度
frame_width  = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num    = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"video frame width:{frame_width} height:{frame_height} frame_num:{frame_num} output:{OUTPUT_VIDEO_PATH}")


# 定义视频编解码器和输出参数  两倍宽度
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (frame_width*2, frame_height)) 


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

    print(origin_image.shape[:2])
    print(f'origin_image shape {origin_image.shape}')
    print(f'mask_image shape {mask_image.shape} dtype {mask_image.dtype}') # shape是(行高,列宽) uint8

    # 放大到原图
    mask_image_resized = cv2.resize( mask_image,
                    dsize=(origin_image.shape[1], origin_image.shape[0]), 
                    interpolation = cv2.INTER_LINEAR)

    mask_image_resized_revert = (255 - mask_image_resized)


    # 归一化 
    mask_image_resized_saved  = mask_image_resized
    mask_image_resized        = mask_image_resized / 255.0 # dtype float64
    mask_image_resized_revert = mask_image_resized_revert / 255.0
    #print(f'mask_image_resized shape {mask_image.shape} dtype {mask_image.dtype}')



    # 替换背景的图片 
    blurred_image = np.zeros(origin_image.shape)
    blurred_image[:,:,0] = 255
    
    # 混合 
    output_image = mask_image_resized_revert * blurred_image + mask_image_resized * origin_image 

    # 合并两个图片 
    print(f" {origin_image.dtype}  {origin_image.shape} {output_image.dtype}  {output_image.shape} ")
    merged_img = cv2.hconcat([origin_image, output_image.astype(np.uint8)])
 
    # 写入视频文件 
    output_video.write(merged_img) 

    # 下一帧
    frame_num_a = frame_num_a + 1 
