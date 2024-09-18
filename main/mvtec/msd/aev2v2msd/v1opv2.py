import lightning as pl
import os
import sys
import time
import torchvision.transforms as transforms

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules.mvtec import MvTecDmV1
from models.mvtec.ae import AeV2V2V1
from models.mvtec.msd import AeV2V2V1MsdV1
from utils import transfer_weights
from utils import init_envir
from utils import load_pre_ae_model
from utils import optuna_main


def main(trial,
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

    monitor = args.monitor + "_auc"
    mse_loss_weight = trial.suggest_float("mse_loss_weight",
                                          0.0002,
                                          1e4,
                                          log=True)
    nu = trial.suggest_float("nu", 1e-4, 1 - 1e-4, log=True)
    lnr_svdd = AeV2V2V1MsdV1(mse_loss_weight=mse_loss_weight,
                             nu=nu,
                             seed=seed,
                             chnum_in=3,
                             objective='soft-boundary')
    transfer_weights(lnr_svdd, auto_enc)
    lnr_svdd.init_center_c(lnr_svdd, datamodule.train_dataloader())
    trainer = pl.Trainer(
        accelerator="gpu",
        # devices=1,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
        # check_val_every_n_epoch=1 if args.visual else epochs,
        default_root_dir=log_path,
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False)
    trainer.fit(model=lnr_svdd, datamodule=datamodule)
    return trainer.callback_metrics[monitor].item()


if __name__ == '__main__':

    start_time = time.perf_counter()
    args = init_envir()
    auto_enc = AeV2V2V1.load_from_checkpoint(
        load_pre_ae_model(
            bash_log_name='bash-logv3',
            # batch_size=args.pre_batch_size,
            batch_size=4,
            radio=args.radio,
            dataset='mvtec',
            n_epochs=args.pre_epochs,
            seed=args.seed,
            normal_class=args.normal_class,
            model_name="aev2v2/v1",
        ))
    datamodule = MvTecDmV1(
        batch_size=args.batch_size,
        seed=args.seed,
        normal_class=args.normal_class,
        transform=transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    )
    optuna_main(args=args, main=main, file=__file__)
    end_time = time.perf_counter()
    # end_time = time.process_time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print("process took %02d:%02d:%02d" % (h, m, s))
