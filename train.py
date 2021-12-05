import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging


from paraphrasegen.constants import BATCH_SIZE, AVAIL_GPUS, MAX_EPOCHS
from paraphrasegen.dataset import CRLDataModule
from paraphrasegen.model import Encoder

from eval import eval


def serialize_params(**kwargs):
    print(kwargs)
    pairs = []
    for key, value in kwargs.items():
        pairs += [f"{key}-{value}"]

    return "_".join(pairs)


def train(
    model_name_or_path,
    task,
    batch_size,
    dims,
    epochs,
    input_mask_rate,
    pooler_type,
    temp,
    hard_negative_weight,
    learning_rate,
    weight_decay,
):
    seed_everything(42)

    run_name = serialize_params(
        model_name_or_path=model_name_or_path,
        task=task,
        batch_size=batch_size,
        dims=dims,
        epochs=epochs,
        input_mask_rate=input_mask_rate,
        pooler_type=pooler_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    print(f"Staring run: {run_name}")

    dm = CRLDataModule(
        model_name_or_path=model_name_or_path,
        task_name=task,
        batch_size=batch_size,
        max_seq_length=32,
    )

    # dm.prepare_data()
    dm.setup("fit")

    encoder = Encoder(
        model_name_or_path,
        input_mask_rate=input_mask_rate,
        mlp_layers=dims,
        pooler_type=pooler_type,
        temp=temp,
        hard_negative_weight=hard_negative_weight,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    eval(encoder)

    swa = StochasticWeightAveraging()
    checkpoint_cb = ModelCheckpoint(monitor="hp_metric", mode="max", save_last=True)

    trainer = Trainer(
        max_epochs=epochs,
        gpus=AVAIL_GPUS,
        log_every_n_steps=2,
        precision=16,
        # logger=TensorBoardLogger("runs/var_mlp_dims"),
        logger=WandbLogger(project="ParaPhraseGen", name=run_name),
        callbacks=[swa, checkpoint_cb],
    )

    trainer.fit(encoder, dm)
    print(f"Best Model: {checkpoint_cb.best_model_path}")
    # trainer.
    eval(encoder)

    # trainer.save_checkpoint(f"checkpoints/{run_name}.ckpt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train an Encoder Model using Contrastive Leanring"
    )

    parser.add_argument("--model_name_or_path", default="roberta-base")
    parser.add_argument("--task", default="paws")
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--epochs", default=MAX_EPOCHS, type=int)
    parser.add_argument("--dims", default=[768], nargs="*", type=int)
    parser.add_argument("--input_mask_rate", default=0.2, type=float)
    parser.add_argument("--pooler_type", default="cls")
    parser.add_argument("--temp", default=0.05, type=float)
    parser.add_argument("--hard_negative_weight", default=0, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    args = parser.parse_args()

    train(
        model_name_or_path=args.model_name_or_path,
        task=args.task,
        batch_size=args.batch_size,
        dims=args.dims,
        epochs=args.epochs,
        input_mask_rate=args.input_mask_rate,
        pooler_type=args.pooler_type,
        temp=args.temp,
        hard_negative_weight=args.hard_negative_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
