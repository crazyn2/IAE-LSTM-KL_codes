import lightning as pl
import os
import sys
import time
import torchvision.transforms as transforms

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
from datamodules.ped2 import Ped2DmV1
from models.ped2.ae import AeV2V2V1
from models.ped2.svdd import AeV2V2V1SdGruV1
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
    datamodule = Ped2DmV1(
        batch_size=batch_size,
        seed=seed,
        time_step=0,
        num_pred=1,
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    )
    lnr_svdd = AeV2V2V1SdGruV1(seed=seed,
                               chnum_in=1,
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
            batch_size=48,
            radio=args.radio,
            dataset='ped2',
            n_epochs=args.pre_epochs,
            seed=args.seed,
            epoch_index=args.pre_batch_size,
            normal_class=None,
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
