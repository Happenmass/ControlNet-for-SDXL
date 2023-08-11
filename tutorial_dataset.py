import json
import cv2
import numpy as np
import einops
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
class SDXLDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./train/train.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['text']


        print('/home/happen/ControlNet/train/' + source_filename)

        source = cv2.imread('/home/happen/ControlNet/train/' + source_filename)
        target = cv2.imread('/home/happen/ControlNet/train/' + target_filename)

        source = cv2.resize(source, (1024, 1024))
        target = cv2.resize(target, (1024, 1024))

        # convert channel from BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # convert channel from HWC to CHW
        source = einops.rearrange(source, 'h w c -> c h w')
        target = einops.rearrange(target, 'h w c -> c h w')

        # normalize source images to [0, 1]
        source = source.astype(np.float32) / 255.5

        # normalize target images to [-1, 1]
        target_norm = target.astype(np.float32) / 255.5
        target_norm = 2.0*target_norm - 1.0
        # target = (target.astype(np.float16)) - 1.0

        return dict(jpg=target_norm, txt=prompt, hint=source, original_size_as_tuple = torch.tensor([1024,1024]), crop_coords_top_left= torch.tensor([0,0]), aesthetic_score=torch.tensor([6.0]), target_size_as_tuple=torch.tensor([1024,1024]))

# class TestDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('/home/happen/ControlNetDataSet/train.json', 'rt') as f:
#             for line in f:
#                 self.data.append(json.loads(line))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#
#         source_filename = item['conditioning_image']
#         target_filename = item['image']
#         prompt = item['text']
#
#         source = cv2.imread('/home/happen/ControlNetDataSet/' + source_filename)
#         target = cv2.imread('/home/happen/ControlNetDataSet/' + target_filename)
#
#         # Do not forget that OpenCV read images in BGR order.
#         source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#         target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
#
#         # Normalize source images to [0, 1].
#         source = source.astype(np.float16) / 255.0
#
#         # Normalize target images to [-1, 1].
#         target = (target.astype(np.float16) / 127.5) - 1.0
#
#         return dict(jpg=target, txt=prompt, hint=source)
