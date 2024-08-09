import numpy as np
import matplotlib.pyplot as plt

# 创建一个时间序列信号
t = np.linspace(0, 1, 500, endpoint=False)

# 创建一个包含两个不同频率的信号
signal = np.sin(50 * 2 * np.pi * t) + 0.5 * np.sin(80 * 2 * np.pi * t)

print(f"t:{t.shape}, signal:{len(signal)} d:{t[1] - t[0]}") # t:(500,), signal:500

# 计算傅里叶变换
fft_result = np.fft.fft(signal)
# 计算频率
frequencies = np.fft.fftfreq(len(signal), d=t[1] - t[0])

# 只取前半部分（因为傅里叶变换结果是对称的）
half_n = len(signal) // 2
fft_magnitude = np.abs(fft_result)[:half_n]
frequencies = frequencies[:half_n]

# 绘制时域信号
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(t, signal)
plt.title('Time Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 绘制频域信号
plt.subplot(122)
plt.stem(frequencies, fft_magnitude, basefmt=" ")
plt.title('Frequency Domain Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()