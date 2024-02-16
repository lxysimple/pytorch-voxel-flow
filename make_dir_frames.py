"""
读一个文件，每一行是一个地址，读取这个地址，并统计该地址下有多少子文件，其数量记录到该地址后面，以空格为分隔符
User
忘记说了，刚刚读取的地址，去掉后面的.avi后缀
"""

import os

# 输入文件路径和输出文件路径
# input_file = "/home/xyli/pytorch-voxel-flow/trainlist01_my.txt"
# output_file = "/home/xyli/pytorch-voxel-flow/trainlist01_nums_my.txt"

# input_file = "/home/xyli/pytorch-voxel-flow/testlist01_my.txt"
# output_file = "/home/xyli/pytorch-voxel-flow/testlist01_nums_my.txt"

input_file = "/home/xyli/pytorch-voxel-flow/trainlist_300vw.txt"
output_file = "/home/xyli/pytorch-voxel-flow/trainlist_nums_300vw.txt"

# 打开输入文件和输出文件
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    # 逐行读取输入文件
    for line in f_in:
        # 去除行尾的换行符，并获取地址
        address = line.strip()
        # 去除地址末尾的 .avi 后缀
        if address.endswith('.avi'):
            address = address[:-4]  # 去除末尾的四个字符（.avi）
        # 统计地址下的子文件数量
        num_files = len(os.listdir(address))
        # 将地址和子文件数量写入输出文件
        f_out.write(f"{address} {num_files}\n")

