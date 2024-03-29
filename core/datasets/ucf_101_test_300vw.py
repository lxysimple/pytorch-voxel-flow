import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from core.utils import transforms as tf

import random

class UCF101Test(Dataset):

    def __init__(self, config):
        super(UCF101Test, self).__init__()
        # dataset_path = 'data/UCF-101_test'
        dataset_path = 'data/ucf-101'
        # with open(os.path.join(dataset_path, config.data_list + '.txt')) as f:
        with open(config.data_list + '.txt') as f:
            self.img_list = []

            # for line in f:
            #     self.img_list.append(line.rstrip().split(' ')[0])
            for line in f:
                video_dir = line.rstrip().split(' ')[0]
                frames_num = int(line.rstrip().split(' ')[1])
                self.img_list.append((video_dir, frames_num))

        self.img_path = dataset_path
        self.config = config

    def __len__(self):
        return len(self.img_list)
    
    # 从一个.pts格式文件中提取68个关键点到列表中，并返回该列表
    def _keypoint_from_pts_(self,file_path):
        # 创建一个列表来存储关键点坐标
        keypoints = []

        with open(file_path, 'r') as file:
            file_content = file.read()

        # 查找花括号内的数据
        start_index = file_content.find('{')  # 找到第一个左花括号的位置
        end_index = file_content.rfind('}')  # 找到最后一个右花括号的位置

        if start_index != -1 and end_index != -1:
            data_inside_braces = file_content[start_index + 1:end_index]  # 提取花括号内的数据

            # 将数据拆分成行
            lines = data_inside_braces.split('\n')
            for line in lines:
                if line.strip():  # 跳过空行
                    x, y = map(float, line.split())  # 假设坐标是空格分隔的
                    keypoints.append(x)
                    keypoints.append(y)
        else:
            print("未找到花括号内的数据")


        return keypoints
    

    # def __getitem__(self, idx, hasmask=True):
    # 从训练代码中看到默认是没有mask提供的
    def __getitem__(self, idx, hasmask=False):

        video_dir = self.img_list[idx][0]
        frames_num = self.img_list[idx][1]
        # frame_idx = random.randint(4, frames_num - 4) # 不同epoch的样本各不相同
        frame_idx = 1

        images = []
        # for i in range(3):
        # for i in range(1,4): # 我生成的帧序号从1开始而不是0
        for i in range(self.config.step):
            
            # img = cv2.imread(
            #     os.path.join(self.img_path, self.img_list[idx],
            #                  'frame_{0:02d}.png'.format(i))).astype(np.float32)

            # img = cv2.imread(
            #     os.path.join(self.img_path, self.img_list[idx], 'img_{0:05d}.jpg'.format(i))).astype(np.float32)          
            
            # img = cv2.imread(
            #             os.path.join(
            #                 video_dir,'{0:06d}.png'.format(frame_idx + i)
            #             )
            #       ).astype(np.float32)
            
            img = self._keypoint_from_pts_( 
                os.path.join(
                            video_dir,'{0:06d}.pts'.format(frame_idx + i)
                        )
            )

            images.append(img)

        # mask = cv2.imread(
        #     os.path.join(self.img_path, self.img_list[idx], 'motion_mask.png'),
        #     cv2.IMREAD_UNCHANGED)
        # mask = (mask.squeeze() > 0).astype(np.uint8)

        
        # # resize
        # target_size = self.config.crop_size
        # images = tf.group_rescale(
        #     images,
        #     0, [cv2.INTER_LINEAR for _ in range(self.config.step)],
        #     dsize=target_size)
        


        
        # norm
        # self.config.input_mean:  [127.5, 127.5, 127.5]
        # self.config.input_std:  [127.5, 127.5, 127.5]
        # print('self.config.input_mean: ', self.config.input_mean)
        # print('self.config.input_std: ', self.config.input_std)

        
        for i in range(3):

            # images[i] = tf.normalize(images[i], self.config.input_mean,
            #                          self.config.input_std)
            
            # images[i] = torch.from_numpy(images[i]).permute(
            #     2, 0, 1).contiguous().float()

            # images[i] = tf.min_max_normalization(images[i])

            images[i] = tf.min_max_normalization_1d(images[i])
            images[i] = torch.from_numpy(np.array(images[i])).float()
            




        # mask = torch.from_numpy(mask).contiguous().long()

        
        

        # if not hasmask:
        #     if self.config.syn_type == 'inter':
        #         return torch.cat([images[0], images[2]], dim=0), images[1]
        #     elif self.config.syn_type == 'extra':
        #         return torch.cat([images[0], images[1]], dim=0), images[2]
        #     else:
        #         raise ValueError('Unknown syn_type ' + self.syn_type)

        # if self.config.syn_type == 'inter':
        #     return torch.cat([images[0], images[2]], dim=0), images[1], mask
        # elif self.config.syn_type == 'extra':
        #     return torch.cat([images[0], images[1]], dim=0), images[2], mask
        # else:
        #     raise ValueError('Unknown syn_type ' + self.syn_type)

        if self.config.syn_type == 'extra':
            images[0] = torch.unsqueeze(images[0], 0)
            images[1] = torch.unsqueeze(images[1], 0)
            images[2] = torch.unsqueeze(images[2], 0)
            return torch.cat([images[0], images[1]], dim=0), images[2]
        else:
            raise ValueError('Unknown syn_type ' + self.syn_type)
