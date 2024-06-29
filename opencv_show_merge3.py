import os 
import cv2 
import subprocess
import sys 
import numpy 

dirLeft  = "/Volumes/MySanDisk/mysdk/bvt_model/HumanSeg_mobile_v2_unet_8_hhl_2023-12-08_Lab53"
dirMid   = "/Volumes/MySanDisk/mysdk/bvt_model/HumanSeg_mobile_v2_unet_8_hhl_2023-12-06_Lab54_add_hand_over_shoulder"
dirRight = "/Volumes/MySanDisk/mysdk/bvt_model/HumanSeg_mobile_v2_unet_8_hhl_2023-12-07_Lab55_add_hand"
#dirGT    = "/Volumes/MySanDisk/mysdk/bvt_model/val_3_mask_clip"
dirGT    = None 

LEFT_NAME = 'Lab53_add_Office'
MID_NAME  = 'Lab54_add_shoulder'
RIGHT_NAME= 'Lab55_add_hand'

TARGET_WIDTH  = 432   # 720
TARGET_HEIGHT = 768   # 1280 

OUTPUT_NAME = f"L_{LEFT_NAME}---M_{MID_NAME}---R_{RIGHT_NAME}"
OUTPUT_VIDEO_PATH = f"/Volumes/MySanDisk/mysdk/bvt_model/compare/{OUTPUT_NAME}_temp.mp4"
OUTPUT_VIDEO_PATH_RESIZE = f"/Volumes/MySanDisk/mysdk/bvt_model/compare/{OUTPUT_NAME}.mp4"

#OUTPUT_VIDEO_PATH = None 

# if OUTPUT_VIDEO_PATH is not None and os.path.exists(OUTPUT_VIDEO_PATH):
#     raise Exception(f"{OUTPUT_VIDEO_PATH} exists")


video_file_list = [
    # "VID_1_out.mp4",
    # "VID_2_out.mp4",
    # "VID_3_out.mp4",
    # "VID_4_out.mp4",
    # "VID_5_out.mp4",
    # "back_chain_on_head_out.mp4",
    # "right_side_black_chain_out.mp4",
    # "long_hair_out.mp4",
    # "show_hand_out.mp4",
    # # #"turn_around_out.mp4"
    # "Chuang_out.mp4",
    # "KeTing_out.mp4",
    # "Office_Road_out.mp4",
    # "Office_Seat_Angle_out.mp4",
    # "Office_Seat_HalfBody_out.mp4",
    # "Office_Seat_Postion_out.mp4",
    # "YangTai_out.mp4",
    "Ly_ByeBye_Long_out.mp4",
    "Ly_Destop_Long_out.mp4",
]


def metric_mse(pred, true):
    #print(f"{pred.shape}, {true.shape}")  # 已经是 (1280, 720), (1280, 720)
    return ((pred - true) ** 2).mean() * 1e3

def iou_pytorch_numpy(preds, target):

    pred_cls   = preds > 0.5
    target_cls = target > 0.5

    intersection = numpy.sum(pred_cls & target_cls)
    union = numpy.sum(pred_cls | target_cls)

    iou = numpy.sum((intersection + 1e-6) / (union + 1e-6)) 

    return iou

def meaning(data):
    return sum(data) / len(data)

def put_text(np_img, text, postion):
    text = text
    org = postion
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 255, 255)
    thickness = 1
    lineType = cv2.LINE_AA
    cv2.putText(np_img, text, org, fontFace, fontScale, color, thickness, lineType)

NUMBER_VIDEOS = len(video_file_list)
QUIT_KEY = ord('q') # 113
LEFT_KEY  = 2
RIGHT_KEY = 3
SPACE_KEY = 32

current_index = 0
end_flag = False 
output_video_writer = None 

while (not end_flag):

    video_file = video_file_list[current_index]

    # 打开源视频 
    left_video   = cv2.VideoCapture(os.path.join(dirLeft,  video_file) )
    middle_video = cv2.VideoCapture(os.path.join(dirMid,   video_file) )
    right_video  = cv2.VideoCapture(os.path.join(dirRight, video_file) )

    # 获取视频帧的宽度和高度
    frame_width  = int(left_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(left_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num    = int(left_video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width_m  = int(middle_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_m = int(middle_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num_m    = int(middle_video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width_r  = int(right_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_r = int(right_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num_r    = int(right_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"file {video_file} ")
    print(f"left_video    frame width:{frame_width}   height:{frame_height}     frame_num:{frame_num}")
    print(f"middle_video  frame width:{frame_width_m}   height:{frame_height_m}   frame_num:{frame_num_m}")
    print(f"right_video   frame width:{frame_width_r}   height:{frame_height_r}   frame_num:{frame_num_r}")

    # 每个视频都清空重新计算 
    iou_list_l = []
    mse_list_l = []
    iou_list_m = []
    mse_list_m = []
    iou_list_r = []
    mse_list_r = []

    gt_mask_video    = None  # GT 
    left_mask_video  = None  # 左mask 
    mid_mask_video   = None  # 中mask 
    right_mask_video = None  # 右mask
    gt_mask_name   = video_file.replace('_out', '' , 1)
    if dirGT is not None and os.path.exists(os.path.join(dirGT, gt_mask_name)):
        gt_mask_path = os.path.join(dirGT, gt_mask_name)
        print(f"gt_mask_path: {gt_mask_path}")

        mask_name = video_file.replace('_out', '_mask' , 1)
        left_mask_path  = os.path.join(dirLeft,  mask_name)  
        mid_mask_path   = os.path.join(dirMid,   mask_name)  
        right_mask_path = os.path.join(dirRight, mask_name)  
        #print(f"left_mask_path: {left_mask_path}")
        
        all_flag = True 
        all_flag = all_flag and os.path.exists(gt_mask_path)  
        all_flag = all_flag and os.path.exists(left_mask_path)  
        all_flag = all_flag and os.path.exists(mid_mask_path)  
        all_flag = all_flag and os.path.exists(right_mask_path)  

        if all_flag :
            gt_mask_video    = cv2.VideoCapture(gt_mask_path)
            left_mask_video  = cv2.VideoCapture(left_mask_path)
            mid_mask_video   = cv2.VideoCapture(mid_mask_path)
            right_mask_video = cv2.VideoCapture(right_mask_path)
    
    while (True):
        ret,  img  = left_video.read()
        ret1, img1 = middle_video.read()
        ret2, img2 = right_video.read() 

        # 读取mask(如果存在)
        gt_mask    = None 
        left_mask  = None 
        mid_mask   = None 
        right_mask = None 
        ret_mask   = True 
        if gt_mask_video is not None:
            mask_ret1, gt_mask = gt_mask_video.read() 
            mask_ret2, left_mask = left_mask_video.read()
            mask_ret3, mid_mask = mid_mask_video.read()
            mask_ret4, right_mask = right_mask_video.read()
            ret_mask = mask_ret1 and mask_ret2 and mask_ret3 and  mask_ret4
            if ret_mask:
                gt_mask    = gt_mask[:, :, 2] / 255.0
                left_mask  = left_mask[:, :, 2] / 255.0
                mid_mask   = mid_mask[:, :, 2] / 255.0
                right_mask = right_mask[:, :, 2] / 255.0 # 归一化 iou是计算>0.5 并且 训练代码计算mse的mask是在0~1

        if not ret or not ret2 or not ret1 or not ret_mask: # 注意 这里没有resize!~~~ TODO
            print(f"end of video read ret={ret} ret1={ret1} ret2={ret2}")
            # 下一个视频
            current_index = current_index + 1 
            if (current_index >= NUMBER_VIDEOS):
                current_index = 0
                # 如果是保存到视频文件 全部文件遍历完就退出 
                if OUTPUT_VIDEO_PATH is not None:
                    end_flag = True 
            break


        img  = cv2.resize(img,  dsize=(TARGET_WIDTH, TARGET_HEIGHT),  interpolation = cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=(TARGET_WIDTH, TARGET_HEIGHT),  interpolation = cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, dsize=(TARGET_WIDTH, TARGET_HEIGHT),  interpolation = cv2.INTER_LINEAR)

        merged_img = cv2.hconcat([img, img1, img2])

        if gt_mask_video is not None: # 计算每帧的mse iou和目前的平均 mse iou, 只判断一个通道 
            iou_l = iou_pytorch_numpy(gt_mask, left_mask)
            iou_m = iou_pytorch_numpy(gt_mask, mid_mask)
            iou_r = iou_pytorch_numpy(gt_mask, right_mask)
            mse_l = metric_mse(gt_mask, left_mask)
            mse_m = metric_mse(gt_mask, mid_mask)
            mse_r = metric_mse(gt_mask, right_mask)
            iou_list_l.append(iou_l)
            iou_list_m.append(iou_m)
            iou_list_r.append(iou_r)
            mse_list_l.append(mse_l)
            mse_list_m.append(mse_m)
            mse_list_r.append(mse_r)
            #print(f"iou:{iou_l:.2f},{iou_m:.2f},{iou_r:.2f};  mse:{mse_l:.2f},{mse_m:.2f},{mse_r:.2f}")
            #print(f"mean iou:{meaning(iou_list_l):.2f},{meaning(iou_list_m):.2f},{meaning(iou_list_r):.2f}")
            #print(f"mean mse:{meaning(mse_list_l):.2f},{meaning(mse_list_m):.2f},{meaning(mse_list_r):.2f}")
            put_text(merged_img, f"cur   iou:{iou_l:.2f}, mse:{mse_l:.2f}",(0 , 50))
            put_text(merged_img, f"mean iou:{meaning(iou_list_l):.2f}, mse:{meaning(mse_list_l):.2f}",(0 , 100)) # y轴需要偏移100
            put_text(merged_img, f"cur   iou:{iou_m:.2f}, mse:{mse_m:.2f}",(img.shape[1] , 50))
            put_text(merged_img, f"mean iou:{meaning(iou_list_m):.2f}, mse:{meaning(mse_list_m):.2f}",(img.shape[1] , 100))
            put_text(merged_img, f"cur   iou:{iou_r:.2f}, mse:{mse_r:.2f}",(img.shape[1] * 2 , 50))
            put_text(merged_img, f"mean iou:{meaning(iou_list_r):.2f}, mse:{meaning(mse_list_r):.2f}",(img.shape[1] * 2 , 100))
            put_text(merged_img, f"{gt_mask_name}", (0 , 150))
            

        if OUTPUT_VIDEO_PATH is not None:
            if output_video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
                print(f"out file size w:{merged_img.shape[1]} h:{merged_img.shape[0]}")
                output_video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (merged_img.shape[1], merged_img.shape[0])) 

            put_text(merged_img, f"{video_file}",  (10                                  , 100))
            put_text(merged_img, f"{LEFT_NAME}" ,  (img.shape[1]//2                     , 50))
            put_text(merged_img, f"{MID_NAME}" ,   (img.shape[1]//2 + img.shape[1]      , 50))
            put_text(merged_img, f"{RIGHT_NAME}" , (img.shape[1]//2 + img.shape[1] * 2  , 50))
            output_video_writer.write(merged_img) 
        else:
            cv2.imshow(OUTPUT_NAME, merged_img)
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
    middle_video.release()
    right_video.release()
    # next 视频 


cv2.destroyAllWindows()

if output_video_writer is not None: # 必须关闭文件 否则后面转换会出错 moov atom not found
    output_video_writer.release()
else:
    sys.exit()

 

# 降低码率 视频文件大小 

# 定义ffmpeg命令
command = ['ffmpeg', '-i', OUTPUT_VIDEO_PATH, '-b:v', '12M' , OUTPUT_VIDEO_PATH_RESIZE]
print(f"reduce video: {' '.join(command)}")

# 执行命令
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 输出结果
print(f"stdout = {result.stdout.decode()}" )
print(f"stderr = {result.stderr.decode()}" )

# 执行成功, 删除大的视频文件 
if result.returncode == 0:
    print(f"reduce ok")
    #os.remove(OUTPUT_VIDEO_PATH)
else:
    print(f"[ERROR] reduce fail")

print(f"reduce done")
