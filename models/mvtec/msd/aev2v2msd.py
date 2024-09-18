from ..base import MvTecMsdV1
from models.base.ae import Ae
from ..ae import EncoderV2
from ..ae import DecoderV2
from torch import nn


class AeV2V2V1MsdV1(MvTecMsdV1):

    def __init__(self,
                 seed,
                 chnum_in=3,
                 mse_loss_weight=1,
                 nu: float = 0.1,
                 lr=1e-4,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(
            seed=seed,
            nu=nu,
            lr=lr,
            mse_loss_weight=mse_loss_weight,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
            objective=objective,
        )
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = EncoderV2(chnum_in, feature_num, feature_num_2,
                                 feature_num_x2)
        self.decoder = DecoderV2(chnum_in, feature_num, feature_num_2,
                                 feature_num_x2)
        self.aucroc_keys = [
            'svdd', 'l1_svdd', 'mse', 'svdd_mse', 'l1_mse', 'l1_svdd_mse'
        ]

    def forward(self, x):
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)

        return Ae(dec_out=dec_x, enc_out=enc_x)


class AeV2V2V1MsdV2(AeV2V2V1MsdV1):

    def __init__(self,
                 seed,
                 chnum_in=3,
                 kl_loss_weight=1,
                 mse_loss_weight=1,
                 nu: float = 0.1,
                 lr=1e-4,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(
            chnum_in=chnum_in,
            seed=seed,
            mse_loss_weight=mse_loss_weight,
            nu=nu,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
            objective=objective,
        )
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")
