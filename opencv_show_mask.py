import cv2
import numpy as np

QUIT_KEY = ord('q') # 113


MASK_VIDEO_PATH="/Volumes/MySanDisk/mysdk/bvt_model/half_body_segmentation_model_quantized_medium_LATEST_LYM_DataCleaning/back_chain_on_head_mask.mp4"
SRC_VIDEO_PATH="/Volumes/MySanDisk/mysdk/bvt_model/back_chain_on_head.mp4"
#MASK_VIDEO_PATH="/Volumes/MySanDisk/mysdk/bvt_model/half_body_segmentation_model_quantized_medium_LATEST_LYM_DataCleaning/right_side_black_chain_mask.mp4"
#SRC_VIDEO_PATH="/Volumes/MySanDisk/mysdk/bvt_model/right_side_black_chain.mp4"
end_flag = False 

def smoothstep(edge0, edge1, x):
    # Scale, bias and saturate x to 0..1 range
    x = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    # Evaluate polynomial
    return x * x * (3 - 2 * x)

while (not end_flag):
    # 打开源视频 
    mask_video  = cv2.VideoCapture(MASK_VIDEO_PATH)
    src_video   = cv2.VideoCapture(SRC_VIDEO_PATH)

    # 获取视频帧的宽度和高度
    mask_width  = int(mask_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask_height = int(mask_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask_num    = int(mask_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"mask width:{mask_width} height:{mask_height} frame_num:{mask_num}")

    src_width   = int(src_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height  = int(src_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_num     = int(src_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"src  width:{src_width}  height:{src_height}   frame_num:{src_num}")

     # 替换背景的图片 
    blurred_image = np.zeros((src_height, src_width, 3))
    blurred_image[:,:,0] = 255


    while mask_video.isOpened():
        ret, mask_image   = mask_video.read()
        ret2, origin_image  = src_video.read() 
        if not ret or not ret2 :
            print(f"end of video read")
            break


        mask_image = cv2.blur(mask_image, (5, 5))

        # 放大到原图
        mask_image_resized = cv2.resize( mask_image,
                        dsize=(origin_image.shape[1], origin_image.shape[0]), 
                        interpolation = cv2.INTER_LINEAR)
        
       
        #  滤波 
        #mask_image_resized = cv2.blur(mask_image_resized, (7, 7))
        #mask_image_resized = cv2.GaussianBlur(mask_image_resized, (11, 11) , 3, 3 )

        #mask_image_resized_revert = (255 - mask_image_resized)

        #  滤波 
        #mask_image_resized_revert = cv2.blur(mask_image_resized_revert, (11, 11))
        #mask_image_resized = (255 - mask_image_resized_revert)



        # 归一化 
        mask_image_resized_saved  = mask_image_resized
        mask_image_resized        = mask_image_resized / 255.0 # dtype float64

        mask_image_resized = smoothstep(0.4, 0.8, mask_image_resized)
        mask_image_resized_revert = (1.0 - mask_image_resized)
        #mask_image_resized_revert = mask_image_resized_revert / 255.0
        #print(f'mask_image_resized shape {mask_image.shape} dtype {mask_image.dtype}')

   
        # 混合 
        output_image = mask_image_resized_revert * blurred_image + mask_image_resized * origin_image 

  
        merged_img = cv2.hconcat([origin_image, output_image.astype(np.uint8)])
        
        cv2.imshow("merged", merged_img)

        key = cv2.waitKey(1) & 0xFF 
        if (key != 255):
            print(f" key = {key}")
        
        if (key == QUIT_KEY):
            end_flag = True 
            break

    mask_video.release()





