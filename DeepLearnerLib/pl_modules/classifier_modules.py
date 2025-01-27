from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics


class ImageClassifier(pl.LightningModule):
    
    def __init__(self,
                 backbone,
                 criterion,
                 learning_rate=1e-3,
                 metrics=["acc", "recall", "f1"],
                 device="cpu"):

        super(ImageClassifier, self).__init__()
        self.save_hyperparameters(ignore=['backbone', 'criterion'])
        self.backbone = backbone
        self.criterion = criterion
        self.learning_rate=learning_rate
        self.train_metrics = {}
        self.val_metrics = {}
        self.metric_names = metrics
        self.is_regression=False
        

        for m in metrics:
            if m == "acc":
                train_metric = torchmetrics.Accuracy(task="binary",average="weighted").to(device)
                val_metric = torchmetrics.Accuracy(task="binary",average="weighted").to(device)
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
            # elif m == "correlation":
            #     train_metric = torchmetrics.PearsonCorrCoef().to(device)
            #     val_metric = torchmetrics.PearsonCorrCoef().to(device)
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

      
        if self.is_regression:
            preds = y_hat.squeeze(-1)  
            y = y.float()  
        else:
            preds = torch.argmax(y_hat, dim=1)  
            y = y.long() 

      
        metrics = {}
        for m, metric in self.train_metrics.items():
            if isinstance(metric, torchmetrics.Metric):
                metric.update(preds, y)  
                metrics[m] = metric.compute() 

    
        self.log('test_loss', loss, prog_bar=True) 
        for m, value in metrics.items():
            self.log(f'test_{m}', value, prog_bar=True) 

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-3
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