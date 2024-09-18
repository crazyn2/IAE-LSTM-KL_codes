import lightning as pl
import os
import sys
from optuna.integration import PyTorchLightningPruningCallback

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules.wtbi import WtbiDmV1
from models.fmnist import AeV4V1
from models.fmnist import AeV4V1MsdLstmV2
from utils import transfer_weights
from utils import init_envir
from utils import optuna_main
from utils import load_pre_ae_model


def lenet_main(trial,
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
    # datamodule = WtbiDmV1(batch_size=batch_size, seed=seed, reload=True)
    kl_loss_weight = trial.suggest_float(
        "kl_loss_weight",
        1e-5,
        1e4,
        # 1,
        log=True)
    nu = trial.suggest_float("nu", 1e-4, 1 - 1e-4, log=True)
    mse_loss_weight = trial.suggest_float("mse_loss_weight",
                                          1e-4,
                                          1e4,
                                          log=True)

    lnr_svdd = AeV4V1MsdLstmV2(kl_loss_weight=kl_loss_weight,
                               mse_loss_weight=mse_loss_weight,
                               nu=nu,
                               seed=seed,
                               objective='soft-boundary')
    transfer_weights(lnr_svdd, auto_enc)
    lnr_svdd.init_center_c(lnr_svdd, datamodule.train_dataloader())
    trainer = pl.Trainer(
        default_root_dir=log_path,
        max_epochs=epochs,
        deterministic=True,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=enable_progress_bar,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=monitor)],
        enable_model_summary=False)

    trainer.fit(model=lnr_svdd, datamodule=datamodule)
    return trainer.callback_metrics[monitor].item()


if __name__ == '__main__':
    args = init_envir()
    datamodule = WtbiDmV1(batch_size=args.batch_size,
                          seed=args.seed,
                          version='v2',
                          reload=True)
    auto_enc = AeV4V1.load_from_checkpoint(
        load_pre_ae_model(bash_log_name='bash-logv3',
                          batch_size=args.batch_size,
                          radio=args.radio,
                          dataset='wtbi',
                          n_epochs=args.pre_epochs,
                          seed=args.seed,
                          normal_class=None,
                          model_name="aev4v1dmv2"))
    optuna_main(args=args, main=lenet_main, file=__file__)
