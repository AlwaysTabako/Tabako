import scipy.io as scio
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

matpath=glob('D:/PythonCode/RF-Diffusion/RF-Diffusion-main/dataset/wifi/output/output/**.mat',recursive=True)

# for path in matpath:
#     # 加载 .mat 文件
#     mat_data = scio.loadmat(path)
#
#     # 打印所有变量名
#     print(mat_data.keys())
#
#     # 查看某个变量的数据，比如 'feature'
#     for feature in mat_data.keys():
#         print(feature)
#         feature_0 = mat_data[feature]
#         print("Feature class:", type(feature_0))
#     print("Feature shape:", mat_data['pred'].shape)
mat = scio.loadmat(matpath[0])
pred = mat['pred']  # shape (1, 512, 90)
# 可视化第0帧、第0维特征
plt.plot(np.abs(pred[0, :, 0]))
plt.title("CSI Feature Channel 0 over Time (512 frames)")
plt.xlabel("Frame index")
plt.ylabel("Amplitude")
plt.show()
