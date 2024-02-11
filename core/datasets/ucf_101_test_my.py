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

    # def __getitem__(self, idx, hasmask=True):
    # 从训练代码中看到默认是没有mask提供的
    def __getitem__(self, idx, hasmask=False):

        video_dir = self.img_list[idx][0]
        frames_num = self.img_list[idx][1]
        frame_idx = random.randint(4, frames_num - 4) # 不同epoch的样本各不相同

        images = []
        # for i in range(3):
        # for i in range(1,4): # 我生成的帧序号从1开始而不是0
        for i in range(self.config.step):
            
            # img = cv2.imread(
            #     os.path.join(self.img_path, self.img_list[idx],
            #                  'frame_{0:02d}.png'.format(i))).astype(np.float32)

            # img = cv2.imread(
            #     os.path.join(self.img_path, self.img_list[idx], 'img_{0:05d}.jpg'.format(i))).astype(np.float32)          
            
            img = cv2.imread(
                        os.path.join(
                            video_dir,'img_{0:05d}.jpg'.format(frame_idx + i)
                        )
                  ).astype(np.float32)
            
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
            images[i] = min_max_normalization(images[i])
            images[i] = torch.from_numpy(images[i]).permute(
                2, 0, 1).contiguous().float()

        # mask = torch.from_numpy(mask).contiguous().long()

        
        

        if not hasmask:
            if self.config.syn_type == 'inter':
                return torch.cat([images[0], images[2]], dim=0), images[1]
            elif self.config.syn_type == 'extra':
                return torch.cat([images[0], images[1]], dim=0), images[2]
            else:
                raise ValueError('Unknown syn_type ' + self.syn_type)

        if self.config.syn_type == 'inter':
            return torch.cat([images[0], images[2]], dim=0), images[1], mask
        elif self.config.syn_type == 'extra':
            return torch.cat([images[0], images[1]], dim=0), images[2], mask
        else:
            raise ValueError('Unknown syn_type ' + self.syn_type)
