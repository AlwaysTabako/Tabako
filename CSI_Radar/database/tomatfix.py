import os
import numpy as np
import scipy.io as scio
import struct
from glob import glob

def read_bf_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()

    cur = 0
    results = []
    while cur + 2 <= len(data):
        try:
            field_len = struct.unpack('H', data[cur:cur + 2])[0]
        except struct.error:
            break  # 剩余数据不足2字节，跳出
        if field_len == 0:
            break
        cur += 2
        if cur + field_len > len(data):
            break  # 避免越界
        field_data = data[cur:cur + field_len]
        cur += field_len

        if len(field_data) < 3:
            continue

        code = field_data[0]
        if code == 187:  # 0xbb = CSI_DATA
            parsed = parse_csi_entry(field_data)
            if parsed is not None:
                results.append(parsed)
    return results

def parse_csi_entry(bytes_data):
    if len(bytes_data) < 25:
        return None

    n_rx = (bytes_data[22] >> 3) & 0x3
    n_tx = (bytes_data[22] >> 5) & 0x7
    n_rx += 1
    n_tx += 1

    payload = bytes_data[23:]
    if n_rx * n_tx * 30 * 2 > len(payload):
        return None

    csi_matrix = np.zeros((n_rx, n_tx, 30), dtype=np.complex64)
    idx = 0
    for sc in range(30):
        for rx in range(n_rx):
            for tx in range(n_tx):
                if idx + 2 > len(payload):
                    return None
                real = struct.unpack('b', payload[idx:idx+1])[0]
                imag = struct.unpack('b', payload[idx+1:idx+2])[0]
                csi_matrix[rx, tx, sc] = complex(real, imag)
                idx += 2
    return {'csi': csi_matrix}

def process_and_save(dat_path):
    frames = read_bf_file(dat_path)
    features = []
    for frame in frames:
        csi = frame['csi']
        if csi.shape[0] < 3 or csi.shape[1] < 3 or csi.shape[2] != 30:
            continue  # 跳过不满足最小维度的帧
        csi = csi[:3, :3, :]  # 裁剪成 (3, 3, 30)
        amp = np.abs(csi).astype(np.float32).flatten()
        features.append(amp[:90])

    if len(features) == 0:
        return False

    feature_mat = np.stack(features, axis=0)  # shape: (N, 90)
    cond_mat = np.zeros((feature_mat.shape[0], 6), dtype=np.float32)

    out_path = os.path.splitext(dat_path)[0] + '.mat'
    scio.savemat(out_path, {
        'feature': feature_mat,
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
    input_dir = './CSI_Gait'
    batch_process(input_dir)