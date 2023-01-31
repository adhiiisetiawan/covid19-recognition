import os
import zipfile
import pytorch_lightning as pl

from typing import Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi


class Covid19DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = 'data/',
        batch_size: int = 32, 
        num_workers: int = 0, 
        pin_memory: bool = False, 
        
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    @property
    def num_classes(self):
        return 3

    def prepare_data(self):
        if os.path.exists('data/covid19-image-dataset.zip'):
            print("Dataset already download")
        else:
            self.api = KaggleApi()
            self.api.authenticate()
            self.api.dataset_download_files('pranavraikokte/covid19-image-dataset', path=self.data_dir)

            with zipfile.ZipFile('data/covid19-image-dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('data/')

    def setup(self, stage=None):
        # build dataset
        self.train = datasets.ImageFolder('data/Covid19-dataset/train')
        self.val = datasets.ImageFolder('data/Covid19-dataset/test')
        self.test = datasets.ImageFolder('data/Covid19-dataset/test')

        self.train.transform = self.augmentation
        self.val.transform = self.transform
        self.test.transform = self.transform
        
    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, 
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = Covid19DataModule()