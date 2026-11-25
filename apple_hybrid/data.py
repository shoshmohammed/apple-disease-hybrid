"""Dataset reading a split CSV (columns: path, label, split[, group_id, leaked])."""
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import IMG_SIZE, AUGMENT

norm = T.Normalize([0.5] * 3, [0.5] * 3)
eval_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])
train_tf = (T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                       T.RandomRotation(15), T.ToTensor(), norm]) if AUGMENT else eval_tf)


class CsvImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df.reset_index(drop=True), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.transform(Image.open(r["path"]).convert("RGB")), int(r["label"])


def load_split(csv_path, data_root=None):
    """Read a split CSV. Paths in the released CSVs are relative to the dataset folder;
    pass --data-root to resolve them on your machine."""
    df = pd.read_csv(csv_path)
    if data_root:
        df["path"] = df["path"].map(lambda p: p if os.path.isabs(p) else os.path.join(data_root, p))
    return df


def class_names(df):
    """Class order used for the labels (alphabetical, as torchvision ImageFolder)."""
    return list(df.drop_duplicates("label").sort_values("label")["class"])
