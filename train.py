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


def train(model_name, task, batch_size, dims, epochs, input_mask_rate, pooler, lr, wd):
    seed_everything(42)

    run_name = serialize_params(
        model_name=model_name,
        task=task,
        batch_size=batch_size,
        dims=dims,
        epochs=epochs,
        input_mask_rate=input_mask_rate,
        pooler=pooler,
        lr=lr,
        wd=wd,
    )

    print(f"Staring run: {run_name}")

    dm = CRLDataModule(
        model_name_or_path=model_name,
        task_name=task,
        batch_size=batch_size,
        max_seq_length=32,
    )

    # dm.prepare_data()
    dm.setup("fit")

    encoder = Encoder(
        model_name,
        input_mask_rate=input_mask_rate,
        mlp_layers=dims,
        pooler_type=pooler,
        learning_rate=lr,
        weight_decay=wd,
    )

    eval(encoder)

    swa = StochasticWeightAveraging()
    checkpoint_cb = ModelCheckpoint(monitor="loss/val", mode="min", save_last=True)

    trainer = Trainer(
        max_epochs=epochs,
        gpus=AVAIL_GPUS,
        log_every_n_steps=2,
        precision=16,
        # logger=TensorBoardLogger("runs/var_mlp_dims"),
        logger=WandbLogger(
            project="ParaPhraseGen", save_dir="wandb_runs/", name=run_name
        ),
        callbacks=[swa, checkpoint_cb],
    )

    trainer.fit(encoder, dm)
    print(f"Best Model: {checkpoint_cb.best_model_path}")
    # trainer.
    eval(encoder)

    trainer.save_checkpoint(f"{run_name}.ckpt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train an Encoder Model using Contrastive Leanring"
    )

    parser.add_argument("--model", "-m", default="roberta-base")
    parser.add_argument("--task", "-t", default="paws")
    parser.add_argument("--batch_size", "-b", default=BATCH_SIZE, type=int)
    parser.add_argument("--epochs", "-e", default=MAX_EPOCHS, type=int)
    parser.add_argument("--dims", "-d", default=[768], nargs="*", type=int)
    parser.add_argument("--input_mask_rate", "-i", default=0.05, type=float)
    parser.add_argument("--pooler", "-p", default="cls")
    parser.add_argument("--lr", "-l", default=3e-5, type=float)
    parser.add_argument("--wd", "-w", default=0.0, type=float)
    args = parser.parse_args()

    train(
        model_name=args.model,
        task=args.task,
        batch_size=args.batch_size,
        dims=args.dims,
        epochs=args.epochs,
        input_mask_rate=args.input_mask_rate,
        pooler=args.pooler,
        lr=args.lr,
        wd=args.wd,
    )
