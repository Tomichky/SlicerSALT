from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics


class ImageClassifier(pl.LightningModule):
    
    def __init__(self,
                 backbone,
                 criterion=nn.CrossEntropyLoss(),
                 learning_rate=1e-3,
                 metrics=["acc", "recall", "f1", "correlation"],
                 device="cpu"):

        super(ImageClassifier, self).__init__()
        self.save_hyperparameters(ignore=['backbone', 'criterion'])
        self.backbone = backbone
        self.criterion = criterion
        self.learning_rate=learning_rate
        self.train_metrics = {}
        self.val_metrics = {}
        self.metric_names = metrics
        
        for m in metrics:
            if m == "acc":
                train_metric = torchmetrics.Accuracy(task="binary").to(device)
                val_metric = torchmetrics.Accuracy(task="binary").to(device)
            elif m == "auc":
                train_metric = torchmetrics.AUROC(pos_label=1, task="binary").to(device)
                val_metric = torchmetrics.AUROC(pos_label=1, task="binary").to(device)
            elif m == "precision":
                train_metric = torchmetrics.Precision(task="binary").to(device)
                val_metric = torchmetrics.Precision(task="binary").to(device)
            elif m == "recall":
                train_metric = torchmetrics.Recall(task="binary").to(device)
                val_metric = torchmetrics.Recall(task="binary").to(device)
            elif m == "f1":
                train_metric = torchmetrics.F1Score(task="binary").to(device)
                val_metric = torchmetrics.F1Score(task="binary").to(device)
            elif m == "correlation":
                train_metric = torchmetrics.PearsonCorrCoef().to(device)
                val_metric = torchmetrics.PearsonCorrCoef().to(device)
            self.train_metrics[m] = train_metric.to(device)
            self.train_metrics[m]=self.train_metrics[m].to(device)
            self.val_metrics[m] = val_metric.to(device)
            self.val_metrics[m]=self.val_metrics[m].to(device)
        

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding
    
    def common_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)  
        y_hat = self.backbone(x).to(self.device)  
        loss = self.criterion(y_hat, y) 

        y_hat = nn.Softmax(dim=-1)(y_hat)[:, 1].to(self.device)

       
        y_hat = y_hat.float()
        y = y.float()

        if mode == "train":
            for m in self.metric_names:
                self.train_metrics[m](y_hat, y)
                self.log(f'train_{m}', self.train_metrics[m].compute(), on_step=True, on_epoch=True)
                self.train_metrics[m].reset()  

        elif mode == "valid":
            for m in self.metric_names:
                self.val_metrics[m](y_hat, y)
                self.log(f'val_{m}', self.val_metrics[m].compute(), on_step=True, on_epoch=True)
                self.val_metrics[m].reset() 

        return loss
    def on_validation_epoch_end(self):
        for m in self.metric_names:
            self.log(f"validation/{m}", self.val_metrics[m].compute())
            self.val_metrics[m].reset()

    def log_scalars(self, scalar_name, scalar_value):
        self.log(scalar_name, scalar_value, on_step=True)

    def training_step(self, batch, batch_idx):
        train_loss = self.common_step(batch, batch_idx, mode="train")
        self.log_scalars('train/train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.common_step(batch, batch_idx, mode="valid")
        self.log_scalars('val_loss', valid_loss)
        return valid_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
       
        preds = torch.argmax(y_hat, dim=1)
        preds=preds.float()
        y=y.float()
        
        metrics = {}
        for m, metric in self.train_metrics.items():
            if isinstance(metric, torchmetrics.Metric):
                metrics[m] = metric(preds, y)
        
        self.log('test_loss', loss, prog_bar=True)
        for m, value in metrics.items():
            self.log(f'test_{m}', value, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default="efficientnet-b0")
        parser.add_argument('--pretrained', action='store_true', default=False)
        return parser




#######################################################################3
##########################FONCTIONNEL DU DEBUT##########################
#######################################################################3
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics


class ImageClassifier(pl.LightningModule):
    
    def __init__(self,
                 backbone,
                 criterion=nn.CrossEntropyLoss(),
                 learning_rate=1e-3,
                 metrics=["acc"],
                 device="cpu"):
        print("Arguments reçus : ", backbone, criterion, learning_rate, metrics, device)

        print("Creation image classifier")
        super(ImageClassifier, self).__init__()
        self.save_hyperparameters(ignore=['backbone', 'criterion'])
        self.backbone = backbone
        self.criterion = criterion
        self.train_metrics = {}
        self.val_metrics = {}
        self.metric_names = metrics
        for m in metrics:
            if m == "acc":
                train_metric = torchmetrics.Accuracy(task="binary").to(device)
                val_metric = torchmetrics.Accuracy(task="binary").to(device)
            elif m == "auc":
                train_metric = torchmetrics.AUROC(pos_label=1, task="binary").to(device)
                val_metric = torchmetrics.AUROC(pos_label=1, task="binary").to(device)
            elif m == "precision":
                train_metric = torchmetrics.Precision(task="binary").to(device)
                val_metric = torchmetrics.Precision(task="binary").to(device)
            elif m == "recall":
                train_metric = torchmetrics.Recall(task="binary").to(device)
                val_metric = torchmetrics.Recall(task="binary").to(device)
            self.train_metrics[m] = train_metric.to(device)
            self.train_metrics[m]=self.train_metrics[m].to(device)
            self.val_metrics[m] = val_metric.to(device)
            self.val_metrics[m]=self.val_metrics[m].to(device)
        print("image classifier cree")
    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def test_epoch_end(self, outputs):
        print("Outputs:", outputs)  
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("test_loss", avg_loss)
        
    def common_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)  
        y_hat = self.backbone(x).to(self.device)  
        loss = self.criterion(y_hat, y) 

       
        y_hat = nn.Softmax(dim=-1)(y_hat)[:, 1].to(self.device)

      
        if mode == "train":
            for m in self.metric_names:
               
                self.train_metrics[m].to(self.device)
                self.train_metrics[m](y_hat, y)
        elif mode == "valid":
            for m in self.metric_names:
            
                self.val_metrics[m].to(self.device)
                self.val_metrics[m](y_hat, y)

        return loss

    def on_validation_epoch_end(self):
        # update and log
        for m in self.metric_names:
            self.log(f"validation/{m}", self.val_metrics[m].compute())
            self.val_metrics[m].reset()

    def log_scalars(self, scalar_name, scalar_value):
        self.log(scalar_name, scalar_value, on_step=True)

    def training_step(self, batch, batch_idx):
        train_loss = self.common_step(batch, batch_idx, mode="train")
        self.log_scalars('train/train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.common_step(batch, batch_idx, mode="valid")
        self.log_scalars('val_loss', valid_loss)
        return valid_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.common_step(batch, batch_idx, mode="test")
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default="efficientnet-b0")
        parser.add_argument('--pretrained', action='store_true', default=False)
        return parser
