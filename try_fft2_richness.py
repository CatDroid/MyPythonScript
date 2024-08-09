import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_image_richness(image):
 
    image_rgb = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # 计算图像的二维傅里叶变换
    f_transform = np.fft.fft2(image)

    # 在计算傅里叶变换后，频谱图的频率分布如下：
    # 未使用 fftshift 前：低频分量位于图像的四角，高频分量位于图像的中心。
    #       左上角（0,0）：表示最低频率分量，即直流分量（DC component）
    #       四角区域：包括(0,0)、(0,N-1)、(M-1,0) 和 (M-1,N-1) 等，表示低频分量。
    # 使用 fftshift 后：低频分量移动到图像的中心，高频分量移动到图像的四周。
    # 这种重新排列使得频谱图在视觉上更直观，因为我们通常更关注中心的低频分量和四周的高频分量。

    # 频率域图像处理
    # http://www.tup.tsinghua.edu.cn/upload/books/yz/099458-01.pdf


    # 移动到图片中心
    f_shift = np.fft.fftshift(f_transform)

    # 中心频率：使用 np.fft.fftshift 后，频谱图的中心位置（通常在图像的中心）对应于图像的直流分量（DC component），即最低频率
    # 高频分量：离中心越远的区域对应图像的高频分量，这些分量表示图像中的快速变化（例如边缘和细节）
    # 对称性：对于实值图像（如灰度图像），频谱图是对称的，因为傅里叶变换结果是共轭对称的

    # 中心附近的区域表示低频分量, 远离中心的区域表示高频分量，对应于图像中变化剧烈的部分，如边缘和纹理
    # 频谱图的
    #   水平方向表示图像的水平频率分量 
    #   垂直方向表示图像的垂直频率分量
    
    # 在二维傅里叶变换的频谱图上，每个点代表一个不同的频率分量，频率在二维空间中是一个向量。对应频率分量的强度和相位


    # 构造高频滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2  # 中心点
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # 半径
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r # 布尔索引
    mask[mask_area] = 0
    
    # 应用高频滤波器
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    image_back = np.fft.ifft2(f_ishift)
    image_back = np.abs(image_back)
    
    # 计算高频分量的能量
    high_freq_energy = np.sum(np.abs(f_shift_filtered) ** 2)
    
    # 评估图像丰富程度  有考虑宽高 
    richness_score = high_freq_energy / (rows * cols)

    richness_score_2 = np.mean(np.abs(f_shift_filtered) ** 2) # 这样算范围更加广 

    richness_score_3 = np.mean(20* np.log( np.abs(f_shift_filtered)  + 1) )

    print(f"Image Richness Score:{richness_score}, {richness_score_2},{richness_score_3}  ({rows},{cols})" )
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(231), plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.axis('off')

    plt.subplot(232), plt.imshow(image, cmap='gray')
    plt.title('Original Gray'), plt.axis('off')
    
    plt.subplot(233), plt.imshow(np.log(np.abs(f_shift)+1), cmap='gray')
    plt.title('FFT of Image'), plt.axis('off')
    
    plt.subplot(234), plt.imshow(mask, cmap='gray')
    plt.title('High Frequency Mask'), plt.axis('off')
    
    plt.subplot(235), plt.imshow(np.log(np.abs(f_shift_filtered)+1), cmap='gray')
    plt.title('Filtered FFT'), plt.axis('off')
    
    plt.subplot(236), plt.imshow(image_back, cmap='gray')
    plt.title('High Frequency Component'), plt.axis('off')
    
    plt.show()
    
    return richness_score

# (360,20)  - (1080,720)
# 读取图像 
image_path = 'image.webp'   # image.shape = (722, 1284, 3)
image = cv2.imread(image_path)
print(f"image.shape = {image.shape}")
offset_x = 450 
image = image[ 30:330, 0+offset_x:300+offset_x , :]
richness_score = compute_image_richness(image)
