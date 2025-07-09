import matplotlib.pyplot as plt
import numpy as np

from wifilib import *
from glob import glob

filepath=[]
filepath = glob('user*.dat', recursive=True)

for path in filepath:
    bf = read_bf_file(path)
    csi_list = list(map(get_scale_csi, bf))
    csi_np = (np.array(csi_list))
    csi_amp = np.abs(csi_np)
    N=int(csi_amp.shape[0])
    csi_amp_flatten = csi_amp.reshape(N,-1)[:,:90]  # 只取前90维

    print(f"csi shape:{path} ", csi_amp.shape,csi_amp_flatten.shape)
    # fig = plt.figure()
    # plt.plot(csi_amp[:, 0, 0, 3])
    # plt.show()