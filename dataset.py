
"""
Jenny Nguyen
April 16, 2026

CS5330 - Final Project: Dental X-Ray Cavity Detection
Dataset loading, label building, and transforms.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BASE_DIR = 'data/archive'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')

IMG_SIZE = 224     # resnet and densenet both expect 224x224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
SEED = 42

# Build image-level labels from CSV
def build_labels(folder):
    """Read the annotations CSV and convert bounding box rows into image-level labels.

    The CSV has one row per bounding box, not per image, so we group by filename
    and check if any row has class == 'Cavity'. If yes -> label 1, else -> label 0.
    """

    df = pd.read_csv(os.path.join(folder, '_annotations.csv'))

    # group by filename, check if any annotation for that image is a cavity
    labels = df.groupby('filename')['class'].apply(
        lambda x: 1 if 'Cavity' in x.values else 0
    ).reset_index()
    labels.columns = ['filename', 'label']

    n_cavity = labels['label'].sum()
    n_none   = (labels['label'] == 0).sum()
    print(f'{os.path.basename(folder)}: {len(labels)} images | {n_cavity} cavity | {n_none} no cavity')
    return labels


# load labels for all three splits
train_labels = build_labels(TRAIN_DIR)
valid_labels = build_labels(VALID_DIR)
test_labels = build_labels(TEST_DIR)


class DentalDataset(Dataset):
    """Custom dataset that loads dental X-ray images from a folder using a labels dataframe."""

    def __init__(self, labels_df, img_dir, transform=None):
        self.labels = labels_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        # convert to RGB because resnet/densenet expect 3-channel input
        # dental X-rays are grayscale but PIL convert handles this
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['label']


# augmentation only on training set to help with the class imbalance
# val/test just get resized and normalized 
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet stats
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])