import lightning as pl
import os
import sys
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules import FMNISTDm
from models.fmnist import AeV4V1
from models.fmnist import AeV4V1MsdLstmV3
from utils import transfer_weights
from utils import init_envir
from utils import load_pre_ae_model


def fmnist_lenet(bash_log_name,
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
    datamodule = FMNISTDm(batch_size=batch_size,
                          radio=radio,
                          seed=seed,
                          normal_class=normal_class,
                          padding=True)
    datamodule.setup('fit')
    auto_enc = AeV4V1.load_from_checkpoint(
        load_pre_ae_model(bash_log_name=bash_log_name,
                          batch_size=batch_size,
                          radio=radio,
                          dataset='fmnist',
                          n_epochs=pre_epochs,
                          seed=seed,
                          normal_class=normal_class,
                          model_name="aev4v1"))

    kl_loss_weights = [
        0.000998, 0.00302, 0.000772, 0.009163, 0.001733, 0.144335, 0.001733,
        0.00001, 0.000052, 0.000406
    ]
    mse_loss_weights = [
        5.112069, 7.980763, 0.18205, 3.237299, 4.066855, 0.000729, 4.066855,
        14.517574, 0.906311, 28.918101
    ]
    nus = [
        0.001147, 0.296802, 0.616333, 0.545526, 0.997169, 0.374395, 0.000269,
        0.354662, 0.000782, 0.011894
    ]
    if objective == 'soft-boundary':
        kl_loss_weights = [
            0.000330, 0.000221, 0.000968, 0.001384, 0.004844, 0.003128,
            0.000022, 0.006297, 0.000015, 0.000049
        ]
        mse_loss_weights = [
            2045.067205, 0.001995, 0.015982, 8.160704, 5.143601, 0.000813,
            0.000107, 254.643766, 0.000147, 0.028476
        ]
    lnr_svdd = AeV4V1MsdLstmV3(seed=seed,
                               kl_loss_weight=kl_loss_weights[normal_class],
                               mse_loss_weight=mse_loss_weights[normal_class],
                               nu=nus[normal_class],
                               objective=objective,
                               visual=args.visual)
    transfer_weights(lnr_svdd, auto_enc)
    lnr_svdd.init_center_c(lnr_svdd, datamodule.train_dataloader())
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        #  enable_checkpointing=False,
        deterministic=True,
        # check_val_every_n_epoch=2,
        check_val_every_n_epoch=1 if args.visual else epochs,
        default_root_dir=log_path,
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False)
    trainer.fit(model=lnr_svdd, datamodule=datamodule)


if __name__ == '__main__':

    start_time = time.perf_counter()
    args = init_envir()
    fmnist_lenet(bash_log_name=args.bash_log_name,
                 normal_class=args.normal_class,
                 pre_epochs=args.pre_epochs,
                 epochs=args.epochs,
                 seed=args.seed,
                 radio=args.radio,
                 batch_size=args.batch_size,
                 enable_progress_bar=args.progress_bar,
                 log_path=args.log_path,
                 objective=args.objective,
                 devices=args.devices)

    end_time = time.perf_counter()
    # end_time = time.process_time()
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print("process took %02d:%02d:%02d" % (h, m, s))
