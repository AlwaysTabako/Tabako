# Wi-Fi CSI Sanitization Pipeline in Python
# Based on the provided MATLAB scripts

import numpy as np
import scipy.io as scio
import os

def set_template(csi_calib, linear_interval):
    """Create CSI calibration template."""
    T, S, A, L = csi_calib.shape
    csi_amp = np.abs(csi_calib)
    csi_phase = np.unwrap(np.angle(csi_calib), axis=1)
    csi_amp_template = np.mean(csi_amp / np.mean(csi_amp, axis=1, keepdims=True), axis=0)  # [1 S A L]

    phase_template = np.zeros_like(csi_calib, dtype=np.float32)
    for p in range(T):
        for a in range(A):
            for e in range(L):
                y = csi_phase[p, linear_interval, a, e]
                x = linear_interval
                coeffs = np.polyfit(x, y, 1)
                y_fit = np.polyval(coeffs, np.arange(S))
                phase_template[p, :, a, e] = csi_phase[p, :, a, e] - y_fit

    csi_phase_template = np.mean(phase_template, axis=0)  # [1 S A L]
    csi_phase_template[linear_interval, :, :] = 0
    return csi_amp_template * np.exp(1j * csi_phase_template)

def nonlinear_calib(csi_src, csi_calib_template):
    """Remove nonlinear phase offset from CSI."""
    csi_amp = np.abs(csi_src)
    csi_phase = np.unwrap(np.angle(csi_src), axis=1)
    csi_unwrap = csi_amp * np.exp(1j * csi_phase)
    return csi_unwrap / csi_calib_template

def rco_calib(csi_calib):
    """Estimate radio chain offset (RCO)."""
    T, S, A, L = csi_calib.shape
    csi_phase = np.unwrap(np.angle(csi_calib), axis=0)
    avg_phase = np.mean(csi_phase[:, :, :, 0], axis=(0, 1))
    return avg_phase - avg_phase[0]

def sto_calib_mul(csi_src):
    """SFO/PDD removal using conjugate multiplication."""
    T, S, A, L = csi_src.shape
    out = np.zeros_like(csi_src, dtype=np.complex64)
    for a in range(A):
        a_next = (a + 1) % A
        out[:, :, a, :] = csi_src[:, :, a, :] * np.conj(csi_src[:, :, a_next, :])
    return out

def sto_calib_div(csi_src):
    """SFO/PDD removal using conjugate division."""
    T, S, A, L = csi_src.shape
    out = np.zeros_like(csi_src, dtype=np.complex64)
    for a in range(A):
        a_next = (a + 1) % A
        out[:, :, a, :] = csi_src[:, :, a, :] / csi_src[:, :, a_next, :]
    return out

def cfo_calib(csi_src, delta_t=4e-6):
    """Estimate CFO (carrier frequency offset)."""
    phase1 = np.angle(csi_src[:, :, :, 0])
    phase2 = np.angle(csi_src[:, :, :, 1])
    phase_diff = np.mean(phase2 - phase1, axis=2)  # [T, S]
    return np.mean(phase_diff / delta_t, axis=1)  # [T]

def agc_calib(csi_src, csi_agc):
    """Remove AGC effect."""
    return csi_src / csi_agc.reshape(-1, 1, 1, 1)

def run_sanitization_pipeline(calib_path, src_path, dst_path, template_path):
    """Complete CSI sanitization pipeline."""
    c = 3e8
    bw = 20e6
    freq = np.linspace(5.8153e9, 5.8347e9, 57)
    lambda_sub = c / freq
    antenna_loc = np.array([[0, 0, 0], [0.0514665, 0, 0], [0, 0.0514665, 0]]).T
    linear_interval = np.arange(20, 39)

    csi_calib = scio.loadmat(calib_path)['csi']
    csi_src = scio.loadmat(src_path)['csi']

    # 生成 template
    csi_template = set_template(csi_calib, linear_interval)
    scio.savemat(template_path, {'csi': csi_template})

    # 非线性校正
    csi_nonlinear = nonlinear_calib(csi_src, csi_template)

    # 去除 SFO/PDD
    csi_clean = sto_calib_div(csi_nonlinear)

    scio.savemat(dst_path, {'csi': csi_clean})
