import os
import torch
import numpy as np
from tfdiff.tools.read_bf_file import read_bf_file

# 替换为你实际的路径
dat_path = './dataset/wifi/raw/test_single.dat'

assert os.path.exists(dat_path), f"{dat_path} 不存在"

# 读取 .dat 文件
csi_list = read_bf_file(dat_path)
print(f"解析得到 {len(csi_list)} 帧 CSI 数据")

features = []

for i, entry in enumerate(csi_list):
    csi = entry['csi']  # shape: (3, 3, 30)
    print(f"[帧 {i}] CSI 形状: {csi.shape}, 数据类型: {csi.dtype}")
    if csi.shape != (3, 3, 30):
        print("跳过此帧，因其格式不正确")
        continue

    amp = np.abs(csi).astype(np.float32)
    amp_vec = amp.reshape(-1)[:90]
    print(f"[帧 {i}] 抽取 90 维 CSI 向量: {amp_vec.shape}")
    features.append(amp_vec)

if len(features) == 0:
    print("❌ 没有有效的 CSI 帧可用于训练，请更换 .dat 文件")
else:
    data_tensor = torch.tensor(np.stack(features), dtype=torch.float32)
    print(f"✅ 最终打包数据张量形状: {data_tensor.shape}")
