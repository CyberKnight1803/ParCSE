from typing import Optional
from pytorch_lightning import trainer

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from constants import AVAIL_GPUS, BATCH_SIZE, PATH_DATASETS


class Encoder(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        input_mask_rate: float = 0.25,
        pooling: str = "mean",
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.input_mask_rate = input_mask_rate
        self.pooling = pooling
        self.bert_model = AutoModel.from_pretrained(
            model_name_or_path, config=self.config
        )

    def forward(self, input_ids, attention_mask):
        sentence_mask = (
            (torch.rand(input_ids.size(), device=self.device) > self.input_mask_rate)
            * (input_ids != 101)
            * (input_ids != 102)
        )

        '''
        Check the effect of using 103 (mask token) vs using the 0 token. 
        No idea how the BERT Model would react to it. 
        Use both with certain percentage amounts? say 80:20
        '''
        input_ids[sentence_mask] = 103  # 103 = The mask token
        # input_ids[sentence_mask] = 0


        '''
        Do we need the rest of the hidden states? idts but then again what do I know.
        head_mask seems interesting since we are masking some parts of the sentences, and _maybe_ we should not pay attention to it. 
        But again, no idea. Check and report.
        '''
        embeddings = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # head_mask=sentence_mask,
            # output_hidden_states=True,
        ).last_hidden_state

        # Pool -- Max pooling, Mean Pooling, CLS Pooling, Dense network? -- Maybe a new Proposal? Based on Convolutional Netowrks?

        if self.pooling == "max":
            pooled = torch.max(
                embeddings, dim=1
            ).values  # torch.max returns max values, and indices.
        elif self.pooling == "mean":
            pooled = torch.mean(embeddings, dim=1)
        elif self.pooling == "cls":
            pooled = embeddings[:, 0]
        elif self.pooling == 'dense':
            pass
        else:
            raise NotImplementedError

        return pooled

    def training_step(self, batch, batch_idx):

        anchor_outputs = self(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
        )

        target_outputs = self(
            input_ids=batch["target_input_ids"],
            attention_mask=batch["target_attention_mask"],
        )

        # fmt: off
        import IPython; IPython.embed()
        import sys; sys.exit();
        # fmt: on

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit":
            return

        train_loader = self.train_dataloader()

        # Required for the learning rate schedule
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

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

    model_name = "roberta-base"

    dm = CRLDataModule(
        model_name_or_path=model_name,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )
    dm.prepare_data()
    dm.setup("fit")
    encoder = Encoder(
        model_name,
        pooling="mean",
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
    )

    trainer = Trainer(max_epochs=1, gpus=AVAIL_GPUS)
    trainer.fit(encoder, dm)
