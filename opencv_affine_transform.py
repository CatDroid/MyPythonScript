import cv2 
import numpy as np
import math 

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # 涉及旋转增强
    # 给定 src 点坐标和绕原点中心点（0,0）的旋转角度的情况下获取目标点坐标
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)


    # 源 3个点(2维坐标)  目标 3个点(2维度坐标)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    # 先算2个点 
    # scale_tmp * shift shift=-0.05~0.05 两个点的位移 不会超过scale_tmp(bbox的宽高)
    # src_w就是scale（bbox宽高, 按目标尺寸按比例缩放后的)  ? dst_dir 是(0, -目标宽/2) 需要确保 宽比高小 ? 
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift  # 也是中心点(向量) + (0, box宽*0.5) 

    # 目标坐标: 第一个点 是 目标分辨率图片的中心点 , 对应 原图的bbox的中心点(加上轻微的位移 scale_tmp * shift )
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir # 等价于 [dst_w * 0.5, (dst_h-dst_w) * 0.5]


    # 第三个点 简单直接返回垂直于a-b的向量 ax*bx = -ay*by 取 bx=-ay by=ax 即(-ay, ax)
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    # dst 是固定的
    # dst = [[72. 96.]
    #        [72. 24.]  24 = 96. - 72.
    #        [ 0. 24.]]

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    # trans = (2, 3)

    return trans

def affine_transform(pt, t):
    # new_pt 2D上的齐次坐标(位置向量)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


img_src = cv2.imread('input.png') # 1080x1920


print(f"w = {img_src.shape[1]}")
print(f"h = {img_src.shape[0]}")
print(f"\n")

fbo_width  = 144
fbo_height = 192
#center = np.array([540, 960])
center = np.array([1080, 960])
scale  = np.array([400, int(400 * fbo_height * 1.0 / fbo_width   )])
rotate_angle = 5  # 框顺时针转30度 图片实际是逆时针转30度
shift = 0.0 # 中心点 在 scale*shift移动

print(f"scale = {scale}\n")

trans = get_affine_transform(center, scale, rotate_angle, [fbo_width, fbo_height], shift=shift)

print(f"trans = \n{trans}, {trans.shape}") # shape = (2, 3)
# 这里会把原图 应用仿射变换 缩小到 H:194 W: 144
img_dst = cv2.warpAffine(img_src, trans, (fbo_width, fbo_height), flags=cv2.INTER_LINEAR)

RS = trans[:,[0,1]]
T  = trans[:,2]

print(f"RS = \n{RS}\n{RS.shape}\nT = \n{T}\n{T.shape}")
print(f"\n")


inv_RS = np.linalg.inv(RS)    # 不能直接传入trans(.shape=(2,3)) # array must be square

inv_T  = - np.dot(inv_RS, T)  # 对于二维数组，np.dot和np.matmul的结果是相同的

print(f"inv_RS = \n{inv_RS}\n{inv_RS.shape}\ninv_T = \n{inv_T}\n{inv_T.shape}")

inv_trans = np.c_[inv_RS, inv_T] # c_ 不会就地操作 不改变 inv_RS

print(f"inv_trans = \n{inv_trans}\n{inv_trans.dtype}\n{inv_trans.shape}")

print(f"\n")
# y = RS * x + t  RS矩阵可逆 t位移向量
# y - t = Rs * x   
# (RS)^-1 * (y - t) = x 
# x = (RS)^-1 * y - ((RS)^-1 * t)   旋转缩放矩阵取逆   位移取反,并且要乘上逆旋转缩放矩阵
# x = S^-1 * R^-1 * y - ((RS)^-1 * t)   也就是变成先做原来的旋转反方向,再缩放原来的倒数
#                                    或者再理解成  (RS)^-1 * (S*S^-1) = (S^-1*R^-1*S^1) * S^-1  也就是缩放是原来的倒数, 但是旋转就不只是原来的逆方向旋转


# 画出变换后的4个角落点 对应原来图上哪个位置
dst_to_src_trans = get_affine_transform(center, scale, rotate_angle, [fbo_width, fbo_height], shift=shift, inv=1)
print(f"dst_to_src_trans =\n{dst_to_src_trans}, {dst_to_src_trans.dtype}, {dst_to_src_trans.shape}")

'''
.---------> x 
| p0     p1
|
|
| p2     p3 
y
'''
p0 = affine_transform([0,           0           ], dst_to_src_trans).astype(np.int32)
p1 = affine_transform([fbo_width,   0           ], dst_to_src_trans).astype(np.int32)
p2 = affine_transform([0,           fbo_height  ], dst_to_src_trans).astype(np.int32)
p3 = affine_transform([fbo_width,   fbo_height  ], dst_to_src_trans).astype(np.int32)


cv2.line(img_src,  p0,  p1,  (0, 0, 255), thickness=2) # 横
cv2.line(img_src,  p0,  p2,  (0, 0, 255), thickness=2) # 竖 

cv2.line(img_src,  p2,  p3,  (255, 0, 0), thickness=2)
cv2.line(img_src,  p1,  p3,  (255, 0, 0), thickness=2)


cv2.imshow('affine_warp', img_dst)
cv2.imshow('src', img_src)
cv2.waitKey(0) 


def cos(angle_in_degrees):
    angle_in_radians = math.radians(angle_in_degrees)
    cos_value = math.cos(angle_in_radians)
    return cos_value

def sin(angle_in_degrees):
    angle_in_radians = math.radians(angle_in_degrees)
    sin_value = math.sin(angle_in_radians)
    return sin_value 


to_center = center.astype(np.float64)
point     = - scale / 2.0  # 左下角 

rotateM = np.array([  # 这个矩阵的正角是逆时针 rotate_angle的定义是顺时针 所以这里角度不用取反, 就已经是逆操作了
            [cos(rotate_angle), -sin(rotate_angle)], 
            [sin(rotate_angle),  cos(rotate_angle)]] , dtype=np.float64)

pointRotated = np.dot(rotateM, point)

pointRotated_Ori = pointRotated + to_center
print(f"pointRotated_Ori = {pointRotated_Ori}")

# 所以 get_affine_transform inv=1 就是从dst往srt的仿射变换---对dst坐标,先缩放再旋转(绕原点)最后平移,得到srt的坐标