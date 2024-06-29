import cv2
import numpy as np

QUIT_KEY = ord('q') # 113


INPUT_VIDEO_PATH="/Volumes/MySanDisk/mysdk/bvt_model/half_body_segmentation_model_quantized_medium_LATEST_LYM_DataCleaning/back_chain_on_head_mask.mp4"

end_flag = False 


while (not end_flag):
    # 打开源视频 
    input_video = cv2.VideoCapture(INPUT_VIDEO_PATH)

    # 获取视频帧的宽度和高度
    frame_width  = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num    = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video frame width:{frame_width} height:{frame_height} frame_num:{frame_num}")

    while input_video.isOpened():
        ret, img = input_video.read()
        if not ret:
            print(f"end of video read")
            break

        dim = (1080, 1920)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)

        resized = cv2.blur(resized, (11, 11))
        #resized = cv2.GaussianBlur(resized, (11, 11) , 3, 3 )

        cv2.imshow("merged", resized)

        key = cv2.waitKey(1) & 0xFF 
        if (key != 255):
            print(f" key = {key}")
        
        if (key == QUIT_KEY):
            end_flag = True 
            break

    input_video.release()





