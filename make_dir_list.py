"""
读目录/home/xyli/data/300vw/edges，统计其中每个一级子文件夹，
每个子文件夹地址写到trainlist_300vw.txt的每一行中
"""

# import os

# # 定义目录路径
# directory = '/home/xyli/data/300VW_Dataset_2015_12_14'
# output = 'trainlist_300vw_68.txt'

# # 打开文件以写入地址
# with open(output, 'w') as f:
#     # 遍历目录中的子文件夹
#     for root, dirs, files in os.walk(directory):
#         # 检查是否有子文件夹
#         if dirs:
#             # 遍历每个子文件夹
#             for d in dirs:
#                 # 获取子文件夹的完整路径
#                 subdir_path = os.path.join(root, d)
#                 # 将子文件夹的地址写入到文件中
#                 f.write(subdir_path + '\n')

import os

# 定义目录路径
directory = '/home/xyli/data/300VW_Dataset_2015_12_14''
output = 'trainlist_300vw_68.txt'

# 获取目录中的一级子文件夹
subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# 写入到 trainlist_300vw.txt 文件中
with open(output, 'w') as f:
    for subdirectory in subdirectories:
        subdir_path = os.path.join(directory, subdirectory, 'annot')
        f.write(subdir_path + '\n')