import lightning as pl
import os
import sys
import time

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from models.cifar10 import AeV1V3MsdLstmV5
from models.cifar10 import AeV1V3
from datamodules import CIFAR10Dm
from utils import transfer_weights
from utils import init_envir
from utils import load_pre_ae_model


def cifar10_lenet(bash_log_name,
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
    cifar10 = CIFAR10Dm(batch_size=batch_size,
                        seed=seed,
                        radio=radio,
                        normal_class=normal_class)
    kl_loss_weight = [
        10.020755, 0.087933, 2.659931, 1.525873, 2.938716, 0.792, 0.645739,
        0.035812, 0.010368, 0.045872
    ]
    mse_loss_weight = [
        29.123988, 0.117254, 3.133786, 2855.311838, 2617.359959, 9.390975,
        64.785139, 1.648697, 0.755313, 26.485921
    ]
    auto_enc = AeV1V3.load_from_checkpoint(
        load_pre_ae_model(bash_log_name=bash_log_name,
                          batch_size=batch_size,
                          radio=radio,
                          n_epochs=pre_epochs,
                          seed=seed,
                          normal_class=normal_class,
                          model_name="aev1v3"))

    lnr_svdd = AeV1V3MsdLstmV5(
        seed=seed,
        # lr_milestone=[100, 150, 250],
        mse_loss_weight=mse_loss_weight[normal_class],
        kl_loss_weight=kl_loss_weight[normal_class],
        objective=objective,
        visual=args.visual)
    transfer_weights(lnr_svdd, auto_enc)
    # lnr_svdd.eval()
    lnr_svdd.init_center_c(lnr_svdd, cifar10.train_dataloader())
    lnr_svdd.train()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        deterministic=True,
        #  check_val_every_n_epoch=1 if args.visual else epochs,
        check_val_every_n_epoch=epochs,
        default_root_dir=log_path,
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False)
    trainer.fit(model=lnr_svdd, datamodule=cifar10)


if __name__ == '__main__':
    start_time = time.perf_counter()
    args = init_envir()
    cifar10_lenet(bash_log_name=args.bash_log_name,
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
