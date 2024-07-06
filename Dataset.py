import torch
import glob
from PIL import Image
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

raf_labels = {'6': 0, '3': 1, '2': 2, '4': 3,
    '7': 4, '5': 5, '1': 6}

jafe_labels = {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3,
    'neutral': 4, 'sadness': 5, 'surprise': 6}

sfew_labels = {'Angry': 0, 'Disgust':1, 'Fear':2, 'Happy':3,
        'Neutral':4, 'Sad':5, 'Surprise':6}

MMAFEDB_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'neutral': 4, 'sad': 5, 'surprise': 6}

oulu_labels = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3,
    'Neutral': 4, 'Sadness': 5, 'Surprise': 6}


class RAF_Paths(Dataset):
    def __init__(self, sour_path, size=224):
        self.sour_path = sour_path
        self.sour_files = glob.glob(self.sour_path+'/*/*')
        self._length = len(self.sour_files)

        self.train_aug = albumentations.Compose([albumentations.Resize(height=size, width=size),
                                                albumentations.RandomCrop(height=size, width=size),
                                                albumentations.Normalize()])

        self.test_aug = albumentations.Compose([albumentations.Resize(height=size, width=size),
                                                albumentations.Normalize()])

    def __len__(self):
        return len(self.sour_files)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.train_aug(image=image)["image"]
        image = image.transpose(2, 0, 1)
        return image

    def get_one_hot(self, path):
        cls = path.split('\\')[-2]
        lab = raf_labels.get(cls)
        one_hot_label = F.one_hot(torch.tensor(int(lab)), num_classes=7) * 1.0
        return one_hot_label

    def __getitem__(self, i):
        example = self.preprocess_image(self.sour_files[i])
        label = self.get_one_hot(self.sour_files[i])
        return example, label

class Target_train_Paths(Dataset):
    def __init__(self, target_path, size=224):
        self.target_path = target_path
        self.target_files = glob.glob(self.target_path+'/*/*')
        self._length = len(self.target_files)

        self.train_aug = albumentations.Compose([albumentations.Resize(height=size, width=size),
                                                albumentations.RandomCrop(height=size, width=size),
                                                albumentations.Normalize()])

    def __len__(self):
        return len(self.target_files)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.train_aug(image=image)["image"]
        image = image.transpose(2, 0, 1)
        return image

    # def get_one_hot(self, path):
    #     cls = path.split('\\')[-2]
    #     if self.mode == 'jafe':
    #         lab = jafe_labels.get(cls)
    #     elif self.mode == 'sfew':
    #         lab = sfew_labels.get(cls)
    #     elif self.mode == 'oulu':
    #         lab = oulu_labels.get(cls)
    #     one_hot_label = F.one_hot(torch.tensor(int(lab)), num_classes=7) * 1.0
    #     return one_hot_label

    def __getitem__(self, i):
        example = self.preprocess_image(self.target_files[i])
        # label = self.get_one_hot(self.target_files[i])
        # return example, label
        return example

class Target_test_Paths(Dataset):
    def __init__(self, target_path, mode, size=224):
        self.target_path = target_path
        self.mode = mode
        self.target_files = glob.glob(self.target_path+'/*/*')
        self._length = len(self.target_files)

        self.test_aug = albumentations.Compose([albumentations.Resize(height=size, width=size),
                                                albumentations.Normalize()])

    def __len__(self):
        return len(self.target_files)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.test_aug(image=image)["image"]
        image = image.transpose(2, 0, 1)
        return image

    def get_one_hot(self, path):
        cls = path.split('\\')[-2]
        if self.mode == 'jafe':
            lab = jafe_labels.get(cls)
        elif self.mode == 'sfew':
            lab = sfew_labels.get(cls)
        elif self.mode == 'oulu':
            lab = oulu_labels.get(cls)
        one_hot_label = F.one_hot(torch.tensor(int(lab)), num_classes=7) * 1.0
        return one_hot_label

    def __getitem__(self, i):
        example = self.preprocess_image(self.target_files[i])
        label = self.get_one_hot(self.target_files[i])
        return example, label

