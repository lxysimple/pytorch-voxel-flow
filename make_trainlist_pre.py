"""
读文件，
/home/xyli/pytorch-voxel-flow/data/ucf-101/annotations/trainlist01.txt
把其第一列加上前缀/home/xyli/pytorch-voxel-flow/data/ucf-101/rawframes/
保存在当前目录下trainlist01_my.txt
"""

# # 输入文件路径
# input_file = "/home/xyli/pytorch-voxel-flow/data/ucf-101/annotations/trainlist01.txt"
# # 输出文件路径
# output_file = "trainlist01_my.txt"

# 输入文件路径
input_file = "/home/xyli/pytorch-voxel-flow/data/ucf-101/annotations/testlist01.txt"
# 输出文件路径
output_file = "testlist01_my.txt"

# 前缀
prefix = "/home/xyli/pytorch-voxel-flow/data/ucf-101/rawframes/"

# 打开输入文件和输出文件
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    # 逐行读取输入文件
    for line in f_in:
        # 去除行尾的换行符，并分割列
        columns = line.strip().split()
        # 如果列数大于等于1，且第一列不为空
        if len(columns) >= 1 and columns[0]:
            # 将第一列加上前缀，写入输出文件
            new_line = prefix + columns[0] + '\n'
            f_out.write(new_line)
