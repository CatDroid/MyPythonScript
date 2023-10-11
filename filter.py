import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.webp', cv2.IMREAD_GRAYSCALE)
print(f"shape {img.shape}")

# 计算x和y方向的梯度
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅度和方向
magnitude = np.sqrt(sobelx**2 + sobely**2)
direction = np.arctan2(sobely, sobelx) * (180 / np.pi) % 180

# 显示结果
cv2.imshow('img', img)
cv2.imshow('Magnitude', magnitude)
cv2.imshow('Direction', direction)
cv2.waitKey(0)
cv2.destroyAllWindows()