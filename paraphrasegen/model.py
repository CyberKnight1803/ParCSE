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


class Pooler(nn.Module):
    """
    Inspired from the SimCSE Paper
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type: str = "cls"):
        super(Pooler, self).__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], f"unrecognized_pooling_type: {self.pooler_type}"

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class MLPLayer(nn.Module):
    def __init__(self, in_dims: int = 768, hidden_dims: int = 768):
        super(MLPLayer, self).__init__()
        self.fc1 = nn.Linear(in_dims, hidden_dims)
        self.layer_norm = nn.LayerNorm(hidden_dims)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        out = self.layer_norm(out)
        return self.activation(out)


class Encoder(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        input_mask_rate: float = 0.1,
        pooler_type: str = "cls",
        learning_rate: float = 3e-5,
        weight_decay: float = 0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.input_mask_rate = input_mask_rate
        self.bert_model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config, cache_dir=PATH_BASE_MODELS
        )

        self.pooler_type = pooler_type
        self.pooler = Pooler(pooler_type)

        self.net = MLPLayer()

        self.loss_fn = ContrastiveLoss()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, do_mlm: bool = True
    ):

        if do_mlm:
            sentence_mask = (
                (
                    torch.rand(input_ids.size(), device=self.device)
                    > self.input_mask_rate
                )
                * (input_ids != 101)
                * (input_ids != 102)
            )

            """
            Check the effect of using 103 (mask token) vs using the 0 token. 
            No idea how the BERT Model would react to it. 
            Use both with certain percentage amounts? say 80:20
            """
            # input_ids[sentence_mask] = 103  # 103 = The mask token
            input_ids[sentence_mask] = 0

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

        pooler_output = self.pooler(attention_mask, bert_outputs)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        # if self.pooler_type == "cls":
        pooler_output = self.net(pooler_output)

        return pooler_output

    def training_step(self, batch, batch_idx):

        # Get Batch of Embeddings: [batch_size, hidden]
        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        loss = self.loss_fn(anchor_outputs, target_outputs)
        self.log("loss/train", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        loss = self.loss_fn(anchor_outputs, target_outputs)

        self.log("loss/val", loss, prog_bar=True)
        self.log("hp_metric", loss)

    def test_step(self, batch, batch_idx):
        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        loss = self.loss_fn(anchor_outputs, target_outputs)

        self.log("loss/test", loss, prog_bar=True)
        self.log("hp_metric", loss)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
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
        )

        return optimizer


if __name__ == "__main__":
    seed_everything(42)

    from dataset import CRLDataModule

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

    trainer = Trainer(
        max_epochs=1,
        gpus=AVAIL_GPUS,
        log_every_n_steps=2,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("runs/"),
    )

    trainer.fit(encoder, dm)
