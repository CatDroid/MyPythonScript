import numpy as np
import cv2
import math
import sys 
import os 
#from PIL import Image, ImageDraw, ImageFont

 

IMAGE_DIR = "/Volumes/MySanDisk/Downloads/VID_20230927_111041"
MASK_DIR  = "/Volumes/MySanDisk/Downloads/VID_20230927_111041_output"
OUTPUT_FILE = "/Volumes/MySanDisk/Downloads/VID_20230927_111041_merge.mp4"

image_itor = sorted(os.listdir(IMAGE_DIR))
mask_itor =  sorted(os.listdir(MASK_DIR))

image_itor = [path for path in image_itor if path[-4:] == ".png" or path[-4:] == ".jpg"]
mask_itor  = [path for path in mask_itor  if path[-4:] == ".png" or path[-4:] == ".jpg"]

if len(image_itor) != len(mask_itor):
    raise Exception(f"file number not match {len(image_itor)} , {len(mask_itor)} ")

#exit(-1) 

 
output_video = None

for i in range(len(image_itor)):
    image_file = os.path.join(IMAGE_DIR, image_itor[i])
    mask_file  = os.path.join(MASK_DIR,  mask_itor[i])
    #print(f"image_file {image_file} {mask_file}")

    origin_image = cv2.imread(image_file)
    mask_image  = cv2.imread(mask_file)

    print(origin_image.shape[:2])
    print(f'origin_image shape {origin_image.shape}')
    print(f'mask_image shape {mask_image.shape} dtype {mask_image.dtype}') # shape是(行高,列宽) uint8


    mask_max_value = np.amax(mask_image)
    if mask_max_value == 1:
        mask_image = mask_image * 255

    # 放大到原图
    if False:
        mask_image_resized = cv2.resize( mask_image,
                        dsize=(origin_image.shape[1], origin_image.shape[0]), 
                        interpolation = cv2.INTER_LINEAR)
    else:
        mask_image_resized = mask_image

    mask_image_resized_revert = (255 - mask_image_resized)


    # 归一化 
    mask_image_resized_saved  = mask_image_resized
    mask_image_resized        = mask_image_resized / 255.0 # dtype float64
    mask_image_resized_revert = mask_image_resized_revert / 255.0
    #print(f'mask_image_resized shape {mask_image.shape} dtype {mask_image.dtype}')


    # 替换背景的图片 
    #blurred_image = cv2.GaussianBlur(origin_image, (125,125), 25)
    blurred_image = np.zeros(origin_image.shape)
    blurred_image[:,:,0] = 255
    
    # 混合 
    output_image = mask_image_resized_revert * blurred_image + mask_image_resized * origin_image # dtype float64

    if True:
        merged_img = cv2.hconcat([origin_image, output_image.astype(np.uint8)])
        if output_video == None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            print(f"output video size {origin_image.shape[1]*2} {origin_image.shape[0]}")
            output_video = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30.0, (origin_image.shape[1]*2, origin_image.shape[0]) ) 
        output_video.write(merged_img) 

    else:    

        # 三图合并成一个图片 
        width1 = origin_image.shape[1]
        width2 = mask_image_resized.shape[1]
        width3 = output_image.shape[1]

        height1 = origin_image.shape[0]
        height2 = mask_image_resized.shape[0]
        height3 = output_image.shape[0]


        result = Image.new('RGBA', (width1 + width2 + width3, max(height1, height2, height3)))
        result.paste(Image.fromarray( cv2.cvtColor(origin_image,                  cv2.COLOR_BGR2RGB)  ), (0, 0))
        result.paste(Image.fromarray( cv2.cvtColor(mask_image_resized_saved,      cv2.COLOR_BGR2RGB)  ), (width1, 0))
        result.paste(Image.fromarray( cv2.cvtColor(np.uint8(output_image),        cv2.COLOR_BGR2RGB)  ), (width1 + width2, 0))

        # /usr/share/fonts/ ubuntu字体库路径 ubuntu不会搜索字体
        fontSize = (int)(origin_image.shape[1]/20)
        setFont = ImageFont.truetype(font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size= fontSize if fontSize > 20 else 20)
        draw = ImageDraw.Draw(result)
        draw.text((20,20), origin_file, font=setFont, fill="#ff0000", direction=None)

   