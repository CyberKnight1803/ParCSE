import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from paraphrasegen.dataset import CRLDataModule
from paraphrasegen.model import Encoder
from paraphrasegen.constants import BATCH_SIZE, AVAIL_GPUS


def train():
    seed_everything(42)

    model_name = "roberta-base"

    dm = CRLDataModule(
        model_name_or_path=model_name,
        batch_size=BATCH_SIZE,
        max_seq_length=32,
    )

    # dm.prepare_data()
    dm.setup("fit")

    encoder = Encoder(model_name, pooler_type="avg_top2")

    checkpoint_cb = ModelCheckpoint(monitor="loss/val", mode="min", save_top_k=2)
    trainer = Trainer(
        max_epochs=1,
        gpus=AVAIL_GPUS,
        log_every_n_steps=10,
        precision=16,
        stochastic_weight_avg=True,
        logger=TensorBoardLogger("runs/"),
        callbacks=[checkpoint_cb],
    )

    trainer.fit(encoder, dm)
    print(f"Best Model: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    train()
