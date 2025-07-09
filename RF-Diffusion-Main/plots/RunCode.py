import os
import subprocess

# 指定文件夹路径
folder_path = 'code'

# 获取所有Python文件的文件名列表
python_files = [f for f in os.listdir(folder_path) if f.endswith('.py')]


python_file=[]
for i in range(len(python_files)+10):
    for filename in python_files:
        if i == int(filename[0:2])-1:
            python_file.append(filename)

print(python_file)
#依次执行每个Python文件
for filename in python_file:
    file_path = os.path.join(folder_path, filename)
    print(f'Running {file_path}...')
    # 使用subprocess运行Python脚本
    subprocess.run(['python', file_path], check=True)