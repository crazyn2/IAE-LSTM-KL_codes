import lightning as pl
import os
import sys
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules import FMNISTDm
from models.fmnist import AeV4V1
from utils import init_envir


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
    cifar10 = FMNISTDm(batch_size=batch_size,
                       seed=seed,
                       radio=radio,
                       normal_class=normal_class,
                    #    gcn=False,
                       padding=True)
    auto_enc = AeV4V1(
        #   [100, 150, 250],
        seed=seed)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        # enable_checkpointing=False,
        deterministic=True,
        # check_val_every_n_epoch=epochs,
        default_root_dir=log_path,
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False)

    trainer.fit(model=auto_enc, datamodule=cifar10)


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
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    print("process took %02d:%02d:%02d" % (h, m, s))
