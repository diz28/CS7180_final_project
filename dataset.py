import torch
from tqdm import tqdm
import time
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
from PIL import Image
import cv2


class MyImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Root directory with two subdirectories 'high_resolution' and 'low_resolution'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.hr_dir = os.path.join(root_dir, 'hr')
        self.lr_dir = os.path.join(root_dir, 'lr')

        self.video_folders = [folder for folder in os.listdir(self.hr_dir) if folder != '.DS_Store']  # Ignoring .DS_Store

        self.frame_pairs = []
        for folder in self.video_folders:
            hr_frames = sorted([file for file in os.listdir(os.path.join(self.hr_dir, folder)) if file != '.DS_Store'])
            lr_frames = sorted([file for file in os.listdir(os.path.join(self.lr_dir, folder)) if file != '.DS_Store'])

            paired_frames = zip(hr_frames, lr_frames)
            for hr, lr in paired_frames:
                self.frame_pairs.append((os.path.join(self.hr_dir, folder, hr), os.path.join(self.lr_dir, folder, lr)))

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.frame_pairs[idx]

        hr_image = cv2.imread(hr_path)
        lr_image = cv2.imread(lr_path)

        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        hr_aug = config.hr_augementation(image=hr_image)["image"]
        lr_aug = config.lr_augementation(image=lr_image)["image"]

        hr_img = config.highres_transform(image=hr_aug)["image"]
        lr_img = config.lowres_transform(image=lr_aug)["image"]        

        return hr_img, lr_img


def test():
    dataset = MyImageFolder(root_dir="data/")
    loader = DataLoader(dataset, batch_size=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
