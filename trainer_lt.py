from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import MathGraphData
from model import GNNTrainer


if __name__ == "__main__":
    # monitoring with wandb
    import wandb
    wandb.login()

    model = GNNTrainer(
        lr=1e-3,
    )
    
    # model = GNNTrainer.load_from_checkpoint(
    #     "math-graph-attention/ey6xvwry/checkpoints/epoch=45-val_seq_acc=0.2932.ckpt",
    #     lr=1e-3,
    #     strict=False,
    # )

    dm = MathGraphData(batch_size=1, workers=1, train_data="data_pkls/los/train.pkl",
                       val_data="data_pkls/los/test2014.pkl",
                       test_data="data_pkls/los/test2016.pkl")

    trainer = Trainer(
        enable_checkpointing=True,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename="{epoch}-{val_link_acc:.4f}",
                save_top_k=5,
                monitor="val_link_acc",
                mode="max",
            ),
            EarlyStopping(
                monitor="val_link_acc",
                patience=5,
                mode="max",
            ),
        ],
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        deterministic=False,
        max_epochs=50,
        log_every_n_steps=50,
        devices=1,
        accelerator='gpu',
        logger=WandbLogger(
            project="math-graph-attention",
            name="egat_crohme2019_link_los_n2e",
        ),
    )

    trainer.fit(model, dm)

    wandb.finish()
