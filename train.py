import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from paraphrasegen.constants import BATCH_SIZE, AVAIL_GPUS, MAX_EPOCHS
from paraphrasegen.dataset import CRLDataModule
from paraphrasegen.model import Encoder


def train(model_name, batch_size, epochs, input_mask_rate, pooler, lr, wd):
    seed_everything(42)

    dm = CRLDataModule(
        model_name_or_path=model_name,
        batch_size=batch_size,
        max_seq_length=32,
    )

    # dm.prepare_data()
    dm.setup("fit")

    encoder = Encoder(
        model_name,
        input_mask_rate=input_mask_rate,
        pooler_type=pooler,
        learning_rate=lr,
        weight_decay=wd,
    )

    checkpoint_cb = ModelCheckpoint(monitor="loss/val", mode="min", save_top_k=2)
    trainer = Trainer(
        max_epochs=epochs,
        gpus=AVAIL_GPUS,
        log_every_n_steps=2,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("runs/"),
        callbacks=[checkpoint_cb],
    )

    trainer.fit(encoder, dm)
    print(f"Best Model: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train an Encoder Model using Contrastive Leanring"
    )

    parser.add_argument("--model", "-m", default="roberta-base")
    parser.add_argument("--batch_size", "-b", default=BATCH_SIZE, type=int)
    parser.add_argument("--epochs", "-e", default=MAX_EPOCHS, type=int)
    parser.add_argument("--input_mask_rate", "-i", default=0.05, type=float)
    parser.add_argument("--pooler", "-p", default="cls")
    parser.add_argument("--lr", "-l", default=3e-5, type=float)
    parser.add_argument("--wd", "-w", default=0.0, type=float)
    args = parser.parse_args()

    print(f"Starting Training with args: {args}")

    train(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        input_mask_rate=args.input_mask_rate,
        pooler=args.pooler,
        lr=args.lr,
        wd=args.wd,
    )
