import scipy.io as scio
from glob import glob
filenames=[]
filenames=glob('CSI_Gait/**/user*.mat',recursive=True)
print(len(filenames))
filename=filenames[::1000]
print(len(filename))
for file in filename:
    mat = scio.loadmat(file)
    print(mat.keys())  # 应该包含 feature 和 cond

    feature = mat['feature']  # shape: (N, 90)
    cond = mat['cond']  # shape: (N, 6)

    print('Feature shape:', feature.shape)
    print('Cond shape:', cond.shape)
    print('Feature dtype:', feature.dtype)
