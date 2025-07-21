import torchvision
import random
from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F


import torchvision.transforms as T

class RandomResizedCropVideo(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=T.InterpolationMode.BICUBIC):
        super(RandomResizedCropVideo, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, frame_list):
        i, j, h, w = self.get_params(frame_list[0], self.scale, self.ratio)
        result=[F.resized_crop(frame, i, j, h, w, self.size, self.interpolation) for frame in frame_list]

        return result

class RandomHorizontalFlipVideo(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipVideo, self).__init__(p)

    def __call__(self, frame_list):
        if torch.rand(1) < self.p:
            return  [F.hflip(frame) for frame in frame_list]
        else:
            return frame_list


class ToTensorVideo(T.ToTensor):
    def __init__(self):
        super(ToTensorVideo, self).__init__()

    def __call__(self, frame_list):
        tensor_video = []
        for frame in frame_list:  # iterate over each frame
            frame = F.to_tensor(frame)  # convert each frame to tensor
            tensor_video.append(frame)
        return torch.stack(tensor_video, dim=0)


class NormalizeVideo(T.Normalize):
    def __init__(self, mean, std):
        super(NormalizeVideo, self).__init__(mean, std)

    def __call__(self, frame_list):
        normalized_video = []
        for frame in frame_list:  # iterate over each frame
            frame = F.normalize(frame, self.mean, self.std)  # normalize each frame
            normalized_video.append(frame)
        return torch.stack(normalized_video, dim=0)  # stack frames back into a video tensor


class ResizeVideo(T.Resize):
    def __init__(self, size):
        super(ResizeVideo, self).__init__(size)

    def __call__(self, frame_list):
        resized_video = []
        for frame in frame_list:  # iterate over each frame
            frame = F.resize(frame, self.size)  # resize each frame
            resized_video.append(frame)
        return resized_video