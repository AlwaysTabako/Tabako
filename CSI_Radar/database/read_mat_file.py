import scipy.io as scio
from glob import glob

matpath=glob('CSI_Gait/20190627/user1/user*.mat',recursive=True)

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
#         # print("Feature example", feature_0[:1])
#     print("Feature shape:", mat_data['feature'].shape)
count=0
for path in matpath:
    mat_data=scio.loadmat(path)
    print(mat_data.keys())
    # if mat_data['feature'].shape[0]==1:
    #     count+=1
    # else:
    #     print(mat_data['feature'].shape)
print(fr'there are {count} 1,90')
