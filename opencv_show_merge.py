import os 
import cv2 

#dirLeft  = "/Volumes/MySanDisk/mysdk/bvt_model/online_model/low_seg/half_body_segmentation_model_quantized_LATEST"
dirLeft  = "/Volumes/MySanDisk/mysdk/bvt_model/HumanSeg_unet_small_zdj_repvit_brance2_dilate_c4_2023-12-20"
dirRight = "/Volumes/MySanDisk/mysdk/bvt_model/HumanSeg_unet_small_zdj_repvit_brance2_dilate_c4_2024-03-13"

video_file_list = [
    "long_hair_girl_1_out.mp4",
    "long_hair_girl_2_out.mp4",

    "long_hair_out.mp4",
    "show_hand_out.mp4"

    "VID_1_out.mp4",
    "VID_2_out.mp4",
    "VID_3_out.mp4",
    "VID_4_out.mp4",
    "VID_5_out.mp4",
    "right_side_black_chain_out.mp4",
    "back_chain_on_head_out.mp4",

    "Chuang_out.mp4",
    "KeTing_out.mp4",
    "Office_Road_out.mp4",
    "Office_Seat_Angle_out.mp4",
    "Office_Seat_HalfBody_out.mp4",
    "Office_Seat_Postion_out.mp4",
    "YangTai_out.mp4"
]


NUMBER_VIDEOS = len(video_file_list)
QUIT_KEY = ord('q') # 113
LEFT_KEY  = 2
RIGHT_KEY = 3
SPACE_KEY = 32

current_index = 0
end_flag = False 

while (not end_flag):

    video_file = video_file_list[current_index]

    # 打开源视频 
    left_video  = cv2.VideoCapture(os.path.join(dirLeft,  video_file) )
    right_video = cv2.VideoCapture(os.path.join(dirRight, video_file) )

    # 获取视频帧的宽度和高度
    frame_width  = int(left_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(left_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num    = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width_r  = int(right_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_r = int(right_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num_r    = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"file {video_file} ")
    print(f"left_video  frame width:{frame_width}   height:{frame_height}   frame_num:{frame_num}")
    print(f"right_video frame width:{frame_width_r} height:{frame_height_r} frame_num:{frame_num_r}")

    while (True):
        ret,  img  = left_video.read()
        ret2, img2 = right_video.read() 

        if not ret or not ret2 :
            print(f"end of video read ret={ret} ret2={ret2}")
            # 下一个视频
            current_index = current_index + 1 
            if (current_index >= NUMBER_VIDEOS):
                current_index = 0
            

            break

        merged_img = cv2.hconcat([img, img2])

        cv2.imshow("merge", merged_img)
       
        ######## key process begin ########
        key = cv2.waitKey(1) & 0xFF 
        if (key != 255):
            print(f" key = {key}")

        if (key == SPACE_KEY):
            while True:
                key = cv2.waitKey(1000) & 0xFF 
                if (key == SPACE_KEY):
                    break
                elif (key == QUIT_KEY):
                    end_flag = True 
                    break
        elif (key == QUIT_KEY):
            end_flag = True 
            break
        elif (key == LEFT_KEY):
            # 上一个视频
            current_index = current_index - 1 
            if (current_index < 0):
                current_index = NUMBER_VIDEOS - 1 
            break
        elif (key == RIGHT_KEY):
            # 下一个视频
            current_index = current_index + 1 
            if (current_index >= NUMBER_VIDEOS):
                current_index = 0
            break 
        ######## key process begin ########

        # end 视频

    left_video.release()
    right_video.release()
    # next 视频 


cv2.destroyAllWindows()
