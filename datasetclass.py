import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform, age_weights=None):
        self.data_dir = data_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.age_weights = age_weights

        self.race_label_map = {
            'East Asian': 0,
            'Indian': 1,
            'Black': 2,
            'White': 3,
            'Middle Eastern': 4,
            'Latino_Hispanic': 5,
            'Southeast Asian': 6
        }

        if self.age_weights is not None:
            self.age_weights = torch.tensor(self.age_weights, dtype=torch.float)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        race_annotation = self.annotations.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        race_annotation_idx = self.race_label_map[race_annotation]
        race_annotation = torch.tensor(race_annotation_idx, dtype=torch.long)

        if self.age_weights is not None:
            age_weight = self.age_weights[idx]
            return image, race_annotation, age_weight
        else:
            return image, race_annotation


class ValDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

        self.race_label_map = {
            'East Asian': 0,
            'Indian': 1,
            'Black': 2,
            'White': 3,
            'Middle Eastern': 4,
            'Latino_Hispanic': 5,
            'Southeast Asian': 6
        }

        self.age_bins = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
        self.age_map = {age: idx for idx, age in enumerate(self.age_bins)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        race_annotation = self.annotations.iloc[idx, 3]
        age_category = self.annotations.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        race_annotation_idx = self.race_label_map[race_annotation]
        race_annotation = torch.tensor(race_annotation_idx, dtype=torch.long)
        age_category_idx = self.age_map[age_category]
        age_category_tensor = torch.tensor(age_category_idx, dtype=torch.long)

        return image, race_annotation, age_category_tensor



