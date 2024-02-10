import os

# 获取当前目录
# current_dir = os.getcwd()
current_dir = '/home/xyli/pytorch-voxel-flow/data/ucf-101/rawframes/'

# 获取当前目录下所有一级子目录
subdirectories = [d for d in os.listdir(current_dir)]

# 打开 result.txt 文件
with open('result.txt', 'w') as f:
    # 遍历每个一级子目录
    for subdir in subdirectories:
        subdir_path = os.path.join(current_dir, subdir)
        # 获取当前一级子目录下所有二级子目录
        sub_subdirectories = [ d for d in os.listdir(subdir_path)]
        # 遍历每个二级子目录
        for sub_subdir in sub_subdirectories:
            sub_subdir_path = os.path.join(subdir_path, sub_subdir)
            # 统计当前二级子目录中图片数量
            image_count = sum([len(files) for _, _, files in os.walk(sub_subdir_path) if any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))) ])
            # 将结果写入 result.txt 文件
            f.write(f"{sub_subdir_path}\t{image_count}\n")
