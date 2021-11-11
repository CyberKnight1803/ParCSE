import time
from typing import Optional

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.profiler import PyTorchProfiler, AdvancedProfiler

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from constants import AVAIL_GPUS, BATCH_SIZE, PATH_DATASETS


class Encoder(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        input_mask_rate: float = 0.25,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        batch_size: int = 32,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.input_mask_rate = input_mask_rate
        self.bert_model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config
        )

        self.loss_fn = nn.CosineEmbeddingLoss(reduction="mean")
        self.net = nn.Sequential(
            nn.Linear(768, 768, bias=False), nn.BatchNorm1d(768), nn.Mish()
        )

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
        embeddings = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # head_mask=sentence_mask,
            # output_hidden_states=True,
        ).last_hidden_state

        # Pool -- Max pooling, Mean Pooling, CLS Pooling, Dense network? -- Maybe a new Proposal? Based on Convolutional Netowrks?
        pooled = embeddings[:, 0]
        return self.net(pooled)

    def training_step(self, batch, batch_idx):

        print(torch.cuda.memory_summary(device=self.device))

        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        loss = self.loss_fn(anchor_outputs, target_outputs, 2 * batch["labels"] - 1)
        self.log("train_loss", loss)

        # fmt: off
        # import IPython; IPython.embed()
        # import sys; sys.exit();
        # fmt: on

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

        loss = self.loss_fn(anchor_outputs, target_outputs, 2 * batch["labels"] - 1)
        self.log("val_loss", loss)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit":
            return

        train_loader = self.train_dataloader()

        # Required for the learning rate schedule
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer


if __name__ == "__main__":
    seed_everything(42)

    from dataset import CRLDataModule

    model_name = "roberta-base"

    dm = CRLDataModule(model_name_or_path=model_name, batch_size=BATCH_SIZE,)
    dm.prepare_data()
    dm.setup("fit")

    torch.cuda.empty_cache()
    encoder = Encoder(model_name, pooling="mean", batch_size=BATCH_SIZE,)

    profiler = PyTorchProfiler(
        dirpath="profiles", filename=f"pytorch-{int(time.time())}"
    )

    trainer = Trainer(
        max_epochs=1,
        gpus=AVAIL_GPUS,
        accumulate_grad_batches=64,
        stochastic_weight_avg=True,
        gradient_clip_val=0.5,
    )
    trainer.fit(encoder, dm)
