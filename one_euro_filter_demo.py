import math
import matplotlib.pyplot as plt 

class LowPassFilter:
    def __init__(self):
        self.x_previous = None

    def __call__(self, x, alpha=0.5):
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered

def get_alpha(rate=30, cutoff=1):
    tau = 1 / (2 * math.pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)

class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):

        # 平滑系数（min_cutoff）：控制信号变化缓慢时的平滑程度。
        # 频率阈值（beta）：控制信号变化较快时的平滑程度

        # 采样频率 freq 指定时间步长或帧率  用于计算滤波器的alpha系数（平滑系数）


        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x):
        if self.dx is None:
            self.dx = 0
        else:
            # (x - self.x_previous) 原始信号的差异 
            self.dx = (x - self.x_previous) * self.freq
        
        # alpha = get_alpha(self.freq, self.dcutoff)
        # print(alpha)
        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        # 保存这一帧的原始值
        self.x_previous = x
        return x_filtered


if __name__ == '__main__':
    filter = OneEuroFilter(freq=15, beta=0.1)
    sensor_data = []
    smoothed_data = []
    for val in range(10):
        x = val + (-1)**(val % 2)
        y = filter(x)
        sensor_data.append(x)
        smoothed_data.append(y)

    plt.figure(figsize=(12, 6))
    plt.plot(sensor_data, label='Sensor Data', color='blue', linestyle='--', marker='o')
    plt.plot(smoothed_data, label='Smoothed Data', color='red', linestyle='-', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Sensor Data vs Smoothed Data')
    plt.legend()
    plt.grid(True)
    plt.show()     