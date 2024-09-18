import lightning as pl
import os
import sys
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules.wtbi import WtbiDmV1
from models.fmnist import AeV4V1
from models.fmnist import AeV4V1MsdLstmV2
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
    datamodule = WtbiDmV1(batch_size=512, seed=seed, reload=True)
    auto_enc = AeV4V1.load_from_checkpoint(
        load_pre_ae_model(bash_log_name='bash-logv3',
                        #   batch_size=args.batch_size,
                          batch_size=128,
                          radio=args.radio,
                          dataset='wtbi',
                          n_epochs=args.pre_epochs,
                          seed=args.seed,
                          normal_class=None,
                          model_name="aev4v1"))

    lnr_svdd = AeV4V1MsdLstmV2(seed=seed,
                               nu=0.898364,
                               mse_loss_weight=0.000158,
                               kl_loss_weight=10e-12,
                               objective=objective)
    transfer_weights(lnr_svdd, auto_enc)
    lnr_svdd.init_center_c(lnr_svdd, datamodule.train_dataloader())
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         enable_checkpointing=False,
                         deterministic=True,
                         num_sanity_val_steps=0,
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
