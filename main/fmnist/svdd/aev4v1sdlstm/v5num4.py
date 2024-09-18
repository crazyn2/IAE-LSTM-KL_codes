import lightning as pl
import os
import sys
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules import FMNISTDm
from models.fmnist import AeV4V1
from models.fmnist import AeV4V1SdLstmV5
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
    # log_path = log_path + datetime.now().strftime('%Y-%m-%d-%H%M%S.%f')[:-3]
    datamodule = FMNISTDm(batch_size=batch_size,
                          radio=radio,
                          seed=seed,
                          normal_class=normal_class,
                          padding=True)
    datamodule.setup("fit")
    auto_enc = AeV4V1.load_from_checkpoint(
        load_pre_ae_model(bash_log_name=bash_log_name,
                          batch_size=batch_size,
                          radio=radio,
                          dataset='fmnist',
                          n_epochs=pre_epochs,
                          seed=seed,
                          normal_class=normal_class,
                          model_name="aev4v1"))

    lnr_svdd = AeV4V1SdLstmV5(
        seed=seed,
        num_layers=4,
        #  lr_milestone=[50, 150, 250],
        objective=objective)
    transfer_weights(lnr_svdd, auto_enc)
    lnr_svdd.init_center_c(lnr_svdd, datamodule.train_dataloader())
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        #  enable_checkpointing=False,
        deterministic=True,
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
    # fmnist_lenet(bash_log_name=args.bash_log_name,
    #              normal_class=args.normal_class,
    #              pre_epochs=args.pre_epochs,
    #              epochs=args.epochs,
    #              seed=args.seed,
    #              radio=args.radio,
    #              batch_size=args.batch_size,
    #              enable_progress_bar=args.progress_bar,
    #              log_path=args.log_path,
    #              objective=args.objective,
    #              devices=args.devices)
