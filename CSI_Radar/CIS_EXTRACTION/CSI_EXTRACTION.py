import numpy as np
import scipy.io as scio
from scipy.constants import c  # 光速
import os
from scipy.fftpack import ifft
from scipy.constants import c  # 光速
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.signal import stft

# 设置路径
src_file_name = './data/csi_src_test.mat'

# 光速（m/s）
c_light = c  # ≈ 3e8

# 带宽（Hz）
bw = 20e6

# 子载波频率（单位 Hz），总共 57 个，中心在 5.825 GHz
subcarrier_freq = np.linspace(5.8153e9, 5.8347e9, 57)

# 子载波波长（λ = c / f）
subcarrier_lambda = c_light / subcarrier_freq  # shape: (57,)

# 三根天线在 AP 上的排列位置（单位米），三根构成一个角形或 L 形阵列
antenna_loc = np.array([
    [0, 0, 0],
    [0.0514665, 0, 0],
    [0, 0.0514665, 0]
]).T  # shape: (3, 3)

# 设置线性相位范围（NIC 类型相关）
linear_interval = np.arange(20, 39)  # 20 到 38 子载波用于相位线性估计

# 读取 CSI 数据
csi_src = scio.loadmat(src_file_name)['csi']  # shape: (T, S, A, L)
T, S, A, L = csi_src.shape


# -----------------------------------
# 示例 1：方向估计（AOA）
# -----------------------------------
def naive_aoa(csi, ant_pos, aoa_ref=np.array([0, 0, 1])):
    """
    简单 AOA 估计方法
    输入:
        csi: CSI 数据 [T, S, A, L]
        ant_pos: 天线位置 [3, A]
    返回:
        aoa: 估计的方向余弦向量 [3,]
    """
    # 取前一帧数据用于估计
    sample_csi = csi[0, :, :, 0]  # shape: (S, A)

    # 提取相位信息
    phase = np.angle(sample_csi)  # shape: (S, A)
    phase_diff = phase[:, 1] - phase[:, 0]  # 用两个天线做差

    # 拟合线性相位，估计角度
    x = subcarrier_freq
    y = phase_diff
    slope, _ = np.polyfit(x, y, deg=1)

    # 简化计算：假设直线传播、简单模型
    aoa_vector = np.array([np.sin(slope), 0, np.cos(slope)])
    aoa_vector = aoa_vector / np.linalg.norm(aoa_vector)
    return aoa_vector


aoa_est = naive_aoa(csi_src, antenna_loc)
aoa_gt = np.array([0, 0, 1])
aoa_error = np.arccos(np.dot(aoa_est, aoa_gt))
print(f"✅ 角度估计误差: {np.rad2deg(aoa_error):.2f} 度")


# -----------------------------------
# 示例 2：TOF（飞行时间）估计
# -----------------------------------
def naive_tof(csi):
    """
    简单 TOF 估计方法
    输入：
        csi: CSI 数据 [T, S, A, L]
    返回：
        tof: 每帧估计的 TOF 时间 [T,]
    """
    T, S, A, L = csi.shape
    tofs = []
    for t in range(T):
        sample = csi[t, :, 0, 0]  # 取第一个天线
        phase = np.unwrap(np.angle(sample))
        freq = subcarrier_freq
        slope, _ = np.polyfit(freq, phase, deg=1)
        tof = -slope / (2 * np.pi)
        tofs.append(tof)
    return np.array(tofs)



def naive_tof(csi_data, bw=20e6):
    """
    粗略估计 ToF（Time-of-Flight）
    输入:
        csi_data: ndarray, shape = [T, S, A, E]
            T: 帧数，S: 子载波数，A: 天线数，E: 扩展字段（通常为1）
        bw: 带宽（Hz），默认 20 MHz
    输出:
        tof_mat: ToF 估计值，单位为秒，shape = [T, A]
    """
    T, S, A, E = csi_data.shape
    ifft_point = 2**int(np.ceil(np.log2(S)))

    # Step 1: IFFT 得到 CIR
    cir_sequence = np.zeros((T, A, E, ifft_point), dtype=np.complex64)
    for t in range(T):
        for a in range(A):
            for e in range(E):
                cir = ifft(csi_data[t, :, a, e], n=ifft_point)
                cir_sequence[t, a, e, :] = cir

    # Step 2: 取 CIR 的前半部分（真实路径）
    half_point = ifft_point // 2
    half_sequence = np.abs(cir_sequence[:, :, :, :half_point])  # 取幅度

    # Step 3: 平均所有 extra（LTF）维度
    cir_avg = np.mean(half_sequence, axis=2)  # [T, A, half_point]

    # Step 4: 找每个 CIR 中的最大峰值索引（对应最大能量点）
    peak_indices = np.argmax(cir_avg, axis=2)  # shape: [T, A]

    # Step 5: 转换为 ToF 值（单位秒）
    tof_mat = peak_indices * S / (ifft_point * bw)  # shape: [T, A]
    return tof_mat

def naive_aoa(csi_data, antenna_loc, est_rco, subcarrier_lambda):
    """
    利用 CSI 做方向估计（AoA）
    Inputs:
        csi_data: ndarray, shape = [T, S, A, E]
        antenna_loc: ndarray, shape = [3, A]，每根天线的空间位置
        est_rco: 射频链偏移，shape = [A, 1]
        subcarrier_lambda: 子载波波长，shape = [S]
    Output:
        aoa_mat: 每帧估计的方向向量，shape = [3, T]
    """
    T, S, A, E = csi_data.shape
    phase = np.unwrap(np.angle(csi_data), axis=1)  # [T, S, A, E]

    # 天线方向差向量 [3, A-1]
    ant_diff = antenna_loc[:, 1:] - antenna_loc[:, [0]]
    ant_len = np.linalg.norm(ant_diff, axis=0)
    ant_dir = ant_diff / ant_len

    # 计算相位差 [T, S, A-1, E]
    phase_diff = phase[:, :, 1:, :] - phase[:, :, [0], :] - est_rco[1:, :].reshape(1, 1, A-1, 1)
    phase_diff = np.unwrap(phase_diff, axis=1)
    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

    # 计算 cos(theta)
    cos_theta = subcarrier_lambda.reshape(1, S, 1, 1) * phase_diff / (2 * np.pi * ant_len.reshape(1, 1, A-1, 1))  # [T, S, A-1, E]
    cos_mean = np.mean(cos_theta, axis=(1, 3))  # [T, A-1]

    # 最小二乘拟合方向向量
    aoa_mat = np.zeros((3, T))
    aoa_init = np.ones(3) / np.sqrt(3)

    for t in range(T):
        def residual(x):
            return ant_dir.T @ x - cos_mean[t]
        result = least_squares(residual, aoa_init, method='lm')
        vec = result.x
        aoa_mat[:, t] = vec / np.linalg.norm(vec)

    return aoa_mat  # [3, T]

def naive_spectrum(csi_data, sample_rate=1000, visable=False):
    """
    利用 CSI 计算 STFT 频谱
    Inputs:
        csi_data: ndarray, shape = [T, S, A, L]
        sample_rate: 采样频率，单位 Hz
        visable: 是否绘图
    Output:
        stft_mag: 频谱图，shape = [frequencies, time]
    """
    # 幅度平方（简化处理）
    amplitude = np.mean(np.abs(csi_data) ** 2, axis=(1, 2, 3))  # [T]

    # STFT
    f, t, Zxx = stft(amplitude, fs=sample_rate, nperseg=128)
    stft_mag = np.abs(Zxx)

    # 可视化
    if visable:
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 20 * np.log10(stft_mag + 1e-6), shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='dB')
        plt.tight_layout()
        plt.show()

    return stft_mag

tof_est = naive_tof(csi_src)
est_dist = np.mean(tof_est) * c_light
print("✅ 实际距离约为: 10 m")
print(f"✅ 估计距离为: {est_dist:.2f} m")
