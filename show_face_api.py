
import numpy as np
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer
from PIL import Image


"""
    直接调用以下代码即可
    from show_edge_api import preprocess
    preprocess(results['img'], results['keypoints'])
"""
def preprocess(image, keypoints, save_path):

    show(image, keypoints, save_path)


def show(image, keypoints, save_path):
    """
        image，是numpy数组
        keypoints，是mumpy数组，[[[1,2],[3,4],...]]，shape=(1,68,2)
        bbox，是mumpy数组，[[30,30, 300, 300]]，shape=(1,4)
    """
    pose_local_visualizer = PoseLocalVisualizer(
                                # line_width = 3,
                                radius=3, 
                                # kpt_color = 'yellow',
                                # link_color = 'yellow'
                                kpt_color = 'white',
                                link_color = 'white'
                            )

    # PoseDataSample存的是所有目标关键点信息
    gt_pose_data_sample = PoseDataSample() 

    # # 将构造的真实关键点存入PoseDataSample中
    gt_instances = InstanceData() # InstanceData对象中存的是一个目标关键点信息
    gt_instances.keypoints = keypoints

    dataset_meta = {
        'skeleton_links': [ # 从0开始
            # 脸轮廓
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
            [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],

            # 左眉毛
            [17, 18], [18, 19], [19, 20], [20, 21],

            # 右眉毛
            [22, 23], [23, 24], [24, 25], [25, 26],

            # 左眼
            [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],  

            # 右眼
            [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],  

            # 嘴巴
            # 外嘴巴
            [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], 
            [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
            # 内嘴巴
            [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67],
            [67, 60],

            # 上鼻子
            [27, 28], [28, 29], [29, 30],

            # 下鼻子
            [31, 32], [32, 33], [33, 34], [34, 35],

        ]
   
                    
    }

    pose_local_visualizer.set_dataset_meta(dataset_meta)
    
    gt_pose_data_sample.gt_instances = gt_instances 
    
   
    # 传入图片、标签、预测、配置，开始画图
    pose_local_visualizer.add_datasample(
                            'image', image,
                            gt_pose_data_sample,
                            out_file = save_path,
                            draw_pred = False,
                            show = False,
                            draw_bbox = True 
                        )