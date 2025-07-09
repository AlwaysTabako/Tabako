from glob import glob
import struct
import os

def read_dat_file(file_name):
    file = open(file_name, 'rb')
    data = file.read()
    result = struct.unpack('H', data)
    file.close()
    return result


dat_files = glob('./CSI_Gait/user1/*.dat')
for file_name in dat_files[::10]:
    data=read_dat_file(file_name)
    print(data)


# def process_and_save(input_path):
#
#
#
#     return result
#
# def batch_process(root_dir):
#     success = 0
#     skipped = 0
#
#     dat_files = glob(os.path.join(root_dir, '**', 'user*.dat'), recursive=True)
#
#     for input_path in dat_files:
#         result = process_and_save(input_path)
#         if result:
#             success += 1
#         else:
#             skipped += 1
#
#     print(f"处理完成：成功 {success} 个文件，跳过 {skipped} 个文件。")
#
# input_dir = './CSI_Gait/20190627/user1'
# batch_process(input_dir)