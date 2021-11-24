from typing import Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler, AdvancedProfiler

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from constants import AVAIL_GPUS, BATCH_SIZE, NUM_WORKERS, PATH_DATASETS


class ContrastiveLoss(nn.Module):
    """
    Loss Function adapted from https://arxiv.org/abs/2104.08821
    """

    def __init__(self, temp: int = 1, eps=1e-20) -> None:
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.eps = eps

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        assert input.shape == target.shape
        assert input.shape[0] == labels.shape[0]

        input_normalized = F.normalize(input)
        target_normalized = F.normalize(target)

        # Efficient Compuation of x_i * x_j' for all i, j in batch_size
        exp_sim_matrix = torch.exp(input_normalized @ target_normalized.T / self.temp)
        exp_loss = (torch.diagonal(exp_sim_matrix) + self.eps) / torch.sum(
            exp_sim_matrix, dim=1
        )

        loss = torch.log(exp_loss)
        return -torch.mean(labels * loss - (1 - labels) * loss)


class Encoder(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        input_mask_rate: float = 0.25,
        embedding_from: str = "single",
        last_n: str = 1,
        pooling: str = "mean",
        use_conv: bool = False,
        learning_rate: float = 3e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.input_mask_rate = input_mask_rate
        self.embedding_from = embedding_from
        self.last_n = last_n
        self.pooling = pooling
        self.use_conv = use_conv  # use a convolutional layer on the sentence embedding?
        self.bert_model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config
        )

        self.bert_embedding_size = (
            self.last_n * 768 if self.embedding_from == "concat" else 768
        )

        if not self.use_conv:
            layers = (
                nn.Linear(self.bert_embedding_size, 768, bias=False),
                nn.BatchNorm1d(768),
                nn.Mish(),  # Try more!
            )
        else:
            layers = (
                nn.Conv2d(1, 6, 64),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 32),
                nn.MaxPool2d(2, 2),
                nn.Linear(42069, 768),  # Determine the magic number later
                nn.BatchNorm1d(768),
                nn.Mish(),  # Try more!
            )

        self.net = nn.Sequential(*layers)
        self.loss_fn = ContrastiveLoss()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit":
            return

        train_loader = self.train_dataloader()

        # Required for the learning rate schedule
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def forward(self, input_ids, attention_mask):
        sentence_mask = (
            (torch.rand(input_ids.size(), device=self.device) > self.input_mask_rate)
            * (input_ids != 101)
            * (input_ids != 102)
        )

        """
        Check the effect of using 103 (mask token) vs using the 0 token. 
        No idea how the BERT Model would react to it. 
        Use both with certain percentage amounts? say 80:20
        """
        input_ids[sentence_mask] = 103  # 103 = The mask token
        # input_ids[sentence_mask] = 0

        """
        Do we need the rest of the hidden states? idts but then again what do I know.
        head_mask seems interesting since we are masking some parts of the sentences, and _maybe_ we should not pay attention to it. 
        But again, no idea. Check and report.
        """
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # head_mask=sentence_mask,
            output_hidden_states=True,
        )

        if self.embedding_from == "last":
            embeddings = bert_outputs.last_hidden_state
        elif self.embedding_from == "single":
            embeddings = bert_outputs.hidden_states[-self.last_n]
        elif self.embedding_from == "sum":
            stacked = torch.stack(bert_outputs.hidden_states[-self.last_n :])
            embeddings = torch.sum(stacked, dim=0)
        elif self.embedding_from == "concat":
            embeddings = torch.cat(bert_outputs.hidden_states[-self.last_n :], dim=2)
        else:
            raise NotImplementedError

        # Pool -- Max pooling, Mean Pooling, CLS Pooling, Dense network? -- Maybe a new Proposal? Based on Convolutional Netowrks?

        if self.pooling == "dense" or self.use_conv:
            pooled = embeddings
        elif self.pooling == "max":
            pooled = torch.max(
                embeddings, dim=1
            ).values  # torch.max returns max values, and indices.
        elif self.pooling == "mean":
            pooled = torch.mean(embeddings, dim=1)
        elif self.pooling == "cls":
            pooled = embeddings[:, 0]
        else:
            raise NotImplementedError

        out = pooled # self.net(pooled)

        return out

    def training_step(self, batch, batch_idx):

        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        # loss = self.loss_fn(anchor_outputs, target_outputs, batch["labels"])
        loss = torch.mean(
            torch.mean(
                F.mse_loss(anchor_outputs, target_outputs, reduction="none"), dim=1
            )
            * (2 * batch["labels"] - 1)
        )
        self.log("train_loss", loss)

        return loss

    def evaluate(self, batch, stage: str = None):
        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        loss = torch.mean(
            torch.mean(
                F.mse_loss(anchor_outputs, target_outputs, reduction="none"), dim=1
            )
            * (2 * batch["labels"] - 1)
        )

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
        if stage == "test":
            self.log("hp_metric", loss)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        loss = torch.mean(
            torch.mean(
                F.mse_loss(anchor_outputs, target_outputs, reduction="none"), dim=1
            )
            * (2 * batch["labels"] - 1)
        )

        self.log(f"val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        loss = torch.mean(
            torch.mean(
                F.mse_loss(anchor_outputs, target_outputs, reduction="none"), dim=1
            )
            * (2 * batch["labels"] - 1)
        )

        self.log(f"test_loss", loss, prog_bar=True)
        self.log("hp_metric", loss)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]  # Remove?
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


if __name__ == "__main__":
    seed_everything(42)

    from dataset import CRLDataModule

    model_name = "distilroberta-base"

    dm = CRLDataModule(
        model_name_or_path=model_name, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )
    dm.prepare_data()
    encoder = Encoder(model_name, pooling="mean", batch_size=BATCH_SIZE,)

    trainer = Trainer(
        max_epochs=1,
        gpus=AVAIL_GPUS,
        log_every_n_steps=10,
        precision=16,
        accumulate_grad_batches=2048 // BATCH_SIZE,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("runs/"),
    )

    trainer.fit(encoder, dm)
