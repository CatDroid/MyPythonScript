import math
import matplotlib.pyplot as plt 
import numpy as np 

class LowPassFilter:
    def __init__(self):
        self.x_previous = None

    # 一阶低通滤波器 指数滤波器
    def __call__(self, x, alpha=0.5):
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered

def get_alpha(rate=30, cutoff=1):
    # 低通滤波器的截止频率
    tau = 1 / (2 * math.pi * cutoff)
    # 采样频率 
    te = 1 / rate
    return 1 / (1 + tau / te)

class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):

        # 直接影响‘位移’低通滤波器的两个超参数
            # 平滑系数（min_cutoff）：控制信号变化缓慢时的平滑程度。（越小，越多频率定义为高频，被‘延迟’和‘衰减’
            # 频率阈值（beta）：控制信号变化较快时的平滑程度

        # 离散信号的采样频率
            # 采样频率 freq 指定时间步长或帧率  用于计算滤波器的alpha系数（平滑系数）

        # 速度滤波器的截止频率(固定) dcutoff 

        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None
        self.__lasttime = None

    def __call__(self, x,  timestamp:float=None):

        # ++++ 根据外部传入的时间戳
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp-self.__lasttime)
        self.__lasttime = timestamp
        # ----

        if self.dx is None:
            self.dx = 0
            # 最开始 认为是没有速度 = 0
        else:
            # (x - self.x_previous) 原始信号的差异 
            # https://github.com/casiez/OneEuroFilter/blob/main/python/OneEuroFilter/OneEuroFilter.py
            # 这个版本用到的是 上一帧修正后的位移
            self.dx = (x - self.x_previous) * self.freq
        
        # alpha = get_alpha(self.freq, self.dcutoff)
        # print(alpha)
        # 速度滤波器 使用固定的截止频率
        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        
        # 平滑后当前的速度 dx_smoothed
        
        # 两个超参数(mincutoff, beta) + 1个当前平滑的速度大小(abs) 来控制 位移低通(指数)滤波器的截止频率
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        # 保存这一帧的原始值
        self.x_previous = x
        return x_filtered


if __name__ == '__main__':

    if False:
        filter = OneEuroFilter(freq=15, beta=0.1)
        sensor_data = []
        smoothed_data = []
        for val in range(10):
            x = val + (-1)**(val % 2)
            y = filter(x)
            sensor_data.append(x)
            smoothed_data.append(y)
    else:


        print("timestamp,noisy,filtered")
        F_Sample = 100.0    # 采样频率  奈奎斯频率=F_Sample/2=100.0/2=50.0(可重建模拟信号50Hz以下,以上会混爹)
        time_length = 10    # 时间长度
 
        # 参数设置
        Freq_sig = 0.2              # (模拟)信号频率
        Ampl_sig  = 1              # (模拟)信号幅度

        # 生成时间序列 步进是 0.01
        time_data = np.arange(0, time_length, 1.0/ F_Sample)

        # 生成正弦波
        sine_wave = Ampl_sig * np.sin(2 * np.pi * time_data * Freq_sig)

        # 添加随机噪音 高斯分布 均值为0 方差为0.1
        #noise_wave = np.random.normal(0, 0.1, len(time_data))
        Ampl_noise = 0.2
        Freq_noise = 10 
        noise_wave = Ampl_noise * np.sin(2 * np.pi * time_data * Freq_noise)
        # 如果噪音的频率 比 采样平率 还要大 就会被过滤掉了 ? 不会 !!!
        # 
        # 根据采样定理,为了准确“重建”信号,采样频率必须至少是信号“最高频率”的两倍
        # 奈奎斯特频率是采样频率的一半
        # 高于奈奎斯特频率的成分会被"折叠"回奈奎斯特频率以下,与原始信号混合
        # (混叠会导致高频成分在采样后变成低频成分，这些低频成分与原始信号混合在一起，难以分离)
        #
        # "抗混叠滤波器anti-aliasing filter"通常用于"模拟信号"。
        # 其主要目的是在信号被数字化之前(进入模数转换器ADC之前)，滤除高于奈奎斯特频率的高频成分，从而防止"混叠"现象的发生。

        # ?? 这里的噪音是指 
        # 奈奎斯特频率以下的噪音：这些噪音在采样过程中不会引起混叠，但仍然会影响信号的质量。可以通过“数字滤波器”来处理和减少这些噪音
        # 
        sine_wave_with_noise = sine_wave + noise_wave


        # 一欧滤波器(一阶低通滤波器 指数滤波)
        config = {
            'freq': F_Sample,       # Hz
            
            'mincutoff': 1.0,       # Hz (位移)高于这个频率的都会被过滤掉 # 慢速抖动 Decreasing the minimum cutoff frequency decreases slow speed jitter
            #'mincutoff': 0.02,     # 比  信号的频率  Freq_sig = 0.2 还要低 就会对原来的信号的幅度(最大位移)衰减 和 滞后
                                    # ???  可以认为 肢体点 的运动频率 不会超过这个 ??? 但是 运动速度 会通过 beta*v 提高最后的f_c 

            'beta': 0.1,             # 速度系数   # Increasing the speed coefficient decreases speed lag. 增加速度系数会减少速度滞后
                                     # 如果物体的速度很高(位移变化幅度比较大)，那么fc会变高, 更高频率信号会通过
            #'beta': 0.0,            #  beta是0 就是一个普通的一阶低通滤波器
                                     # 'mincutoff': 0.02 'beta:0.0' 就会被信号(Freq_sig = 0.2)本省都过滤掉了
            
            'dcutoff': 1.0          # 固定的速度截止频率
            }
        f = OneEuroFilter(**config)

        # 'mincutoff': 1.0,
        # Freq_noise = 10  相当于 截止频率的10倍 按照一阶低通滤波的幅频响应 应该下降了20db 幅度值是原来的0.1 --> Ampl_noise = 0.2 再 * 0.1 = 0.02

        sensor_data = []
        smoothed_data = []
        for timestamp, raw_data in zip(time_data, sine_wave_with_noise) :
            filtered = f(raw_data, timestamp)
            sensor_data.append(raw_data)
            smoothed_data.append(filtered)
            

    plt.figure(figsize=(20, 8))
    plt.plot(time_data, sensor_data, label='Sensor Data',  color='red', linestyle='-', marker='.' ) # marker='x'  marker='o'
    plt.plot(time_data, smoothed_data, label='Smoothed Data', color='blue', linestyle='--', marker='.')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Sensor Data vs Smoothed Data')
    plt.legend()
    plt.grid(True)
    plt.show()     