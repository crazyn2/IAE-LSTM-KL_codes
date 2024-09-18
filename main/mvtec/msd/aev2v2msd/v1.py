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


def main(bash_log_name,
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
    datamodule = MvTecDmV1(
        batch_size=batch_size,
        seed=seed,
        normal_class=normal_class,
        transform=transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    )
    nus = [
        0.203559, 0.975294, 0.000275, 0.798075, 0.049296, 0.004556, 0.033248,
        0.046456, 0.000328, 0.821166, 0.001784, 0.001548, 0.020619, 0.005574,
        0.001143
    ]
    mse_loss_weights = [
        0.073546, 0.038967, 0.017701, 936.012256, 14.315515, 2.083169,
        284.489828, 0.000582, 0.008332, 998.22267, 3548.381346, 48.562379,
        0.098499, 5554.917392, 218.942727
    ]
    lnr_svdd = AeV2V2V1MsdV1(seed=seed,
                             nu=nus[normal_class],
                             mse_loss_weight=mse_loss_weights[normal_class],
                             chnum_in=3,
                             objective=objective,
                             visual=args.visual)
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
    main(bash_log_name='bash-logv3',
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
