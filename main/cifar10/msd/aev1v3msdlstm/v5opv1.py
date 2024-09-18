import lightning as pl
import os
import sys
from optuna.integration import PyTorchLightningPruningCallback

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from models.cifar10 import AeV1V3MsdLstmV5
from models.cifar10 import AeV1V3
from datamodules import CIFAR10Dm
from utils import transfer_weights
from utils import init_envir
from utils import load_pre_ae_model
from utils import optuna_main


def cifar10_lenet(trial,
                  bash_log_name,
                  normal_class,
                  seed,
                  pre_epochs,
                  epochs,
                  log_path,
                  objective,
                  radio,
                  batch_size,
                  devices=2,
                  enable_progress_bar=False):
    monitor = args.monitor + "_roc_auc_sk"
    kl_loss_weight = trial.suggest_float(
        "kl_loss_weight",
        1e-5,
        1e4,
        # 1,
        log=True)
    mse_loss_weight = trial.suggest_float("mse_loss_weight",
                                          0.0002,
                                          1e4,
                                          log=True)

    lnr_svdd = AeV1V3MsdLstmV5(kl_loss_weight=kl_loss_weight,
                               mse_loss_weight=mse_loss_weight,
                               seed=seed,
                               objective='one-class')
    transfer_weights(lnr_svdd, auto_enc)
    lnr_svdd.init_center_c(lnr_svdd, cifar10.train_dataloader())
    lnr_svdd.train()
    trainer = pl.Trainer(
        # accelerator="gpu",
        # devices=1,
        num_sanity_val_steps=0,
        default_root_dir=log_path,
        max_epochs=epochs,
        deterministic=True,
        enable_checkpointing=False,
        #  check_val_every_n_epoch=4,
        enable_progress_bar=enable_progress_bar,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=monitor)],
        enable_model_summary=False)

    trainer.fit(model=lnr_svdd, datamodule=cifar10)
    return trainer.callback_metrics[monitor].item()


if __name__ == '__main__':
    args = init_envir()
    cifar10 = CIFAR10Dm(batch_size=args.batch_size,
                        seed=args.seed,
                        radio=args.radio,
                        normal_class=args.normal_class)
    auto_enc = AeV1V3.load_from_checkpoint(
        load_pre_ae_model(bash_log_name=args.bash_log_name,
                          batch_size=args.batch_size,
                          radio=args.radio,
                          n_epochs=args.pre_epochs,
                          seed=args.seed,
                          normal_class=args.normal_class,
                          model_name="aev1v3"))
    optuna_main(args=args, main=cifar10_lenet, file=__file__)
