import lightning as pl
import os
import sys
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules.wtbi import WtbiDmV1
from models.fmnist import AeV2V2V1
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
    datamodule = WtbiDmV1(batch_size=batch_size, seed=seed)
    auto_enc = AeV2V2V1(seed=seed)
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         num_sanity_val_steps=0,
                         deterministic=True,
                         default_root_dir=log_path,
                         max_epochs=epochs,
                         enable_progress_bar=enable_progress_bar,
                         enable_model_summary=False)

    trainer.fit(model=auto_enc, datamodule=datamodule)


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
