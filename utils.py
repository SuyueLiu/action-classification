import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import torch


def get_label(csv_path, img_name):
    data_pair = pd.read_csv(csv_path)
    label = data_pair[data_pair['image_name'] == img_name]['label'].to_numpy()

    return label


class ImgLabelLoader(Dataset):
    def __init__(self, img_dir, csv_path):
        super(ImgLabelLoader, self).__init__()
        self.transformer = transforms.Compose([
            transforms.RandomResizedCrop(300, scale=(0.2, 1), ratio=(0.75, 1)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.5, 0.5, 0.5],
            #     std=[0.5, 0.5, 0.5]
            # ),
        ])
        self.img_dir = img_dir
        self.csv_path = csv_path

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        img = self.transformer(img)
        label = get_label(self.csv_path, img_name)
        label = label.squeeze()

        return img, label

    def __len__(self):
        self.img_names = os.listdir(self.img_dir)
        if '.DS_Store' in self.img_names:
            self.img_names.remove('.DS_Store')

        return len(self.img_names)
