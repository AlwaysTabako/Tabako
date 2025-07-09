from glob import glob
import numpy as np
import os
from wifilib import *
import scipy.io as scio

import re

def extract_cond_from_filename(filename):
    # 假设格式为 userX-Y-Z-rN.dat
    m = re.match(r'user(\d+)-(\d+)-(\d+)-r(\d+)\.dat', os.path.basename(filename))
    if m:
        # 解析为 int 并补齐到6维
        cond = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), 0, 0]
    else:
        cond = [0, 0, 0, 0, 0, 0]
    return cond

def process_and_save(filepath):
    bf = read_bf_file(filepath)
    csi_list = list(map(get_scale_csi, bf))
    csi_np = (np.array(csi_list))
    csi_amp = np.abs(csi_np)
    N = int(csi_amp.shape[0])
    csi_amp_flatten = csi_amp.reshape(N, -1)[:, :90]  # 只取前90维
    feature = np.array(csi_amp_flatten)
    cond_info = extract_cond_from_filename(filepath)
    cond_mat = np.tile(cond_info, (feature.shape[0], 1)).astype(np.float32)
    out_path = os.path.splitext(filepath)[0] + '.mat'
    scio.savemat(out_path, {
        'feature': feature,
        'cond': cond_mat
    })
    return True

def batch_process(root_dir):
    success = 0
    skipped = 0

    dat_files = glob(os.path.join(root_dir, '**', 'user*.dat'), recursive=True)

    for input_path in dat_files:
        result = process_and_save(input_path)
        if result:
            success += 1
        else:
            skipped += 1

    print(f"处理完成：成功 {success} 个文件，跳过 {skipped} 个文件。")

if __name__ == '__main__':
    # 替换为你自己的路径
    input_dir = 'D:/PythonCode/RF-Diffusion/20190627'
    batch_process(input_dir)