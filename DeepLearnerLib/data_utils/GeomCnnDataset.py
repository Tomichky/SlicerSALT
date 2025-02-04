import logging
from typing import Optional
from operator import itemgetter

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, RandRotate, RandFlip, RandZoom, EnsureType,LoadImage
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from DeepLearnerLib.data_utils.utils import get_image_files_single_scalar
from DeepLearnerLib.data_utils.CustomDataset import GeomCnnDataset


class GeomCnnDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = -1,
                 val_frac: float = 0.2,
                 num_workers=4,
                 data_tuple=None,
                 file_paths=None):
        super(GeomCnnDataModule, self).__init__()
        self.w = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.FILE_PATHS = file_paths
        self.train_transforms = Compose(
            [
             	LoadImage(image_only=True),
                EnsureChannelFirst(channel_dim='no_channel'),
                NormalizeIntensity(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandFlip(spatial_axis=0, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                EnsureType(),
            ]
	)
	
        self.val_transforms = Compose(
                [LoadImage(image_only=True), EnsureChannelFirst(channel_dim='no_channel'), NormalizeIntensity(), EnsureType()]
            )
        self.test_transform = Compose(
                    [LoadImage(image_only=True), EnsureChannelFirst(channel_dim='no_channel'), NormalizeIntensity(), EnsureType()]
                )
        self.data_tuple = data_tuple
        self.save_hyperparameters()


    def setup(self, stage: Optional[str] = None):
        print("Setting up data loaders ...")
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            print("UUUUUUUU",self.FILE_PATHS)
            train_files, train_labels = get_image_files_single_scalar(FILE_PATHS=self.FILE_PATHS,data_dir="TRAIN_DATA_DIR")
            length = len(train_files)
            if self.batch_size == -1:
                self.batch_size = length
            if self.data_tuple is None:
                train_x, val_x, \
                train_y, val_y = \
                    train_test_split(train_files, train_labels,
                                     test_size=self.val_frac, shuffle=True,
                                     stratify=train_labels, random_state=42)
                self.train_ds = GeomCnnDataset(train_x, train_y, self.train_transforms)
                self.val_ds = GeomCnnDataset(val_x, val_y, self.val_transforms)
                logging.info(f"Training samples: {len(train_y)}, Validation samples: {len(val_y)}")
            else:
                self.train_ds = GeomCnnDataset(self.data_tuple[0], self.data_tuple[1], self.train_transforms)
                self.val_ds = GeomCnnDataset(self.data_tuple[2], self.data_tuple[3], self.val_transforms)
                logging.info(f"Training samples: {len(self.data_tuple[0])}, Validation samples: {len(self.data_tuple[2])}")

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            test_files, test_labels = get_image_files_single_scalar(FILE_PATHS=self.FILE_PATHS,data_dir="TEST_DATA_DIR")
            self.test_ds = GeomCnnDataset(test_files, test_labels, self.test_transform)
        print("Finished loading !!!")


    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size)


class GeomCnnDataModuleKFold:
    def __init__(self,
                 batch_size,
                 num_workers,
                 n_splits=2,
                 file_paths=None):
        super(GeomCnnDataModuleKFold, self).__init__()
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.num_workers = num_workers
        self.FILE_PATHS = file_paths
        
        self.datamodules = self.split_data()

    def split_data(self):
        print("VVVVVVVVVV",type(self.FILE_PATHS))
        train_files, train_labels = get_image_files_single_scalar(FILE_PATHS=self.FILE_PATHS,data_dir="TRAIN_DATA_DIR")
        skf = StratifiedKFold(n_splits=self.n_splits)
        datamodule_list = []
        for train_index, val_index in skf.split(train_files, train_labels):
            train_x = list(itemgetter(*train_index.tolist())(train_files))
            train_y = list(itemgetter(*train_index.tolist())(train_labels))
            valid_x = list(itemgetter(*val_index.tolist())(train_files))
            valid_y = list(itemgetter(*val_index.tolist())(train_labels))
            datamodule_list.append(GeomCnnDataModule(
                batch_size=self.batch_size,
                data_tuple=(train_x, train_y, valid_x, valid_y),
                num_workers=self.num_workers))
        return datamodule_list

    def get_folds(self):
        return self.datamodules


logger = TensorBoardLogger(save_dir="logs/", name="geom_cnn_training")





