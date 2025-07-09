import scipy.io as scio
from glob import glob

mat_path=glob('D:/PythonCode/RF-Diffusion/20190627/user**.mat',recursive=True)
for path in mat_path:
    mat_data=scio.loadmat(path)
    print(mat_data.keys())
    feature = mat_data['feature']  # shape: (N, 90)
    cond = mat_data['cond']  # shape: (N, 6)
    print('Feature shape:', feature.shape)
    print('Cond shape:', cond.shape)
    print('Feature dtype:', feature.dtype)