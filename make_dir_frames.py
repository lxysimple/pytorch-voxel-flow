import os

# 获取当前目录
# current_dir = os.getcwd()
current_dir = ''

# 获取当前目录下所有子目录
subdirectories = [d for d in os.listdir(current_dir) if os.path.isdir(d)]

# 打开 result.txt 文件
with open('result.txt', 'w') as f:
    # 遍历每个子目录
    for subdir in subdirectories:
        subdir_path = os.path.join(current_dir, subdir)
        # 统计当前子目录中图片数量
        image_count = sum([len(files) for _, _, files in os.walk(subdir_path)])
        # 将结果写入 result.txt 文件
        f.write(f"{subdir_path}\t{image_count}\n")
