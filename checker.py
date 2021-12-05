from os import stat_result
from typing import Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoConfig, AutoModel, AdamW

from paraphrasegen.loss import ContrastiveLoss
from paraphrasegen.constants import (
    AVAIL_GPUS,
    BATCH_SIZE,
    PATH_BASE_MODELS,
)

from paraphrasegen.model import Encoder, Pooler, MLPLayer
from paraphrasegen.loss import Similarity

class Checker(pl.LightningModule):
    def __init__(
        self, 
        encoder: nn.Module, 
        pooler_type: str = "cls",
        learning_rate: float = 3e-4,
        weight_decay: float = 0,
        in_dims: int = 768
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore="encoder")
        self.encoder = encoder 
        self.net = nn.Sequential(
            nn.Linear(in_dims, 1),
            nn.Sigmoid()
        )

        self.sim = nn.CosineSimilarity()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outs = self.encoder(input_ids, attention_mask)
        return outs 
    
    def training_step(self, batch, batch_idx):
        # Get Batch of Embeddings: [batch_size, hidden]
        
        labels = batch['labels']

        anchor_outs= self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outs= self(
                input_ids=batch["target_input_ids"],
                attention_mask=batch["target_attention_mask"],
        )

        outs = self.sim(target_outs, anchor_outs)
        outs = self.net(outs)

        loss = self.loss_fn(labels.float(), outs.float())
        self.log("loss/train", loss)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Get Batch of Embeddings: [batch_size, hidden]
        
        labels = batch['labels']

        anchor_outs= self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outs= self(
                input_ids=batch["target_input_ids"],
                attention_mask=batch["target_attention_mask"],
        )

        outs = self.sim(target_outs, anchor_outs)


        loss = self.loss_fn(labels.float(), outs.float())
        self.log("loss/val", loss)

        return loss

    def test_step(self, batch, batch_idx):
        # Get Batch of Embeddings: [batch_size, hidden]
        
        labels = batch['labels']

        anchor_outs= self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outs= self(
                input_ids=batch["target_input_ids"],
                attention_mask=batch["target_attention_mask"],
        )

        outs = self.sim(target_outs, anchor_outs)


        loss = self.loss_fn(labels.float(), outs.float())
        self.log("loss/test", loss)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        return optimizer
    

if __name__=="__main__":
    seed_everything(42)
    from paraphrasegen.dataset import CRLDataModule

    model_name = "roberta-base"
    dm = CRLDataModule(
        model_name_or_path=model_name,
        batch_size=BATCH_SIZE,
        max_seq_length=32,
        # padding="do_not_pad",
    )
    # dm.prepare_data()
    dm.setup("fit")

    encoder = Encoder(model_name, pooler_type="avg_top2")
    checker = Checker(encoder)

    trainer = Trainer(
        max_epochs=1,
        gpus=AVAIL_GPUS,
        log_every_n_steps=10,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("classification_runs/"),
    )

    trainer.fit(checker, dm)
    trainer.test(checker, dm)