from ..ae import EncoderLstmV8
from models.base.sd import BaseSd


class AeV8SdLstmV1(BaseSd):
    def __init__(self,
                 seed,
                 rep_dim=64,
                 center=None,
                 nu: float = 0.1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestone=[50],
                 optimizer_name='amsgrad',
                 log_red=False,
                 objective='one-class'):
        super().__init__(seed, center, nu, rep_dim, lr, weight_decay,
                         lr_milestone, optimizer_name, log_red, objective)
        self.encoder = EncoderLstmV8(rep_dim)

    def forward(self, x):

        enc_x = self.encoder(x)

        return {
            # "dec_out": dec_x,
            'enc_out': enc_x
        }
