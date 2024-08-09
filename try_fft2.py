import numpy as np
import matplotlib.pyplot as plt
import cv2 


image_path = "image.webp"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

print(f"w: {image.shape[1]}, h: {image.shape[0]}")

image = image[ 30:710, 340:1080 , :]

print(f"crop image {image.shape[1]},{image.shape[0]}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"gray image {gray.shape}")


# 复数数组，表示图像的频域表示，其中数组中的每个元素包含一个复数，这个复数代表特定频率成分的振幅和相位
f = np.fft.fft2(gray)
print(f"fft2 .shape = {f.shape}") # fft2 .shape = (200, 200) 跟属于二维图片(像素序列一样的宽高)
print(f"fft2 .dtype = {f.dtype}") # complex128
print(f"type fft2[0,0] = { type(f[0,0]) }") # 类型是 numpy.complex128  float 64bit的实数 float 64bit的虚数


# 提取实部、虚部、幅度和相位
#real_part = np.real(fft_result)
#imag_part = np.imag(fft_result)
#magnitude = np.abs(fft_result)
#phase = np.angle(fft_result)


# 将傅里叶变换结果移动到中心
fshift = np.fft.fftshift(f)  

 # 计算幅度谱
#magnitude_spectrum = 20*np.log(np.abs(fshift)) # 使用 np.abs 计算傅里叶变换结果的幅度 magnitude
magnitude_spectrum = 20*np.log(np.abs(fshift) + 1) # 幅度谱并进行对数缩放，使得结果更易于可视化
high_freq_density = np.mean(magnitude_spectrum)
print(f"high_freq_density = {high_freq_density}")

plt.figure(figsize=(12, 6))
plt.subplot(121)
#plt.imshow(gray, cmap='gray')
plt.imshow(  cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])
plt.show()
