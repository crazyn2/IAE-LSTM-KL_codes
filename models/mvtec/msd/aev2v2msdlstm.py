from models.mvtec.base import MvTecMsdV1
import torch
from utils import get_radius
from torch import nn
import torch.nn.functional as F
from models.base.ae import Ae
from models.mvtec.ae import EncoderV2
from models.mvtec.ae import DecoderV2


class AeV2V2V1MsdLstmV1(MvTecMsdV1):

    def __init__(self,
                 chnum_in,
                 seed,
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
        self.lstm = torch.nn.LSTM(256 * 4 * 4, 256 * 4 * 4, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        self.aucroc_keys = [
            'svdd', 'l1_svdd', 'mse', 'svdd_mse', 'l1_mse', 'l1_svdd_mse'
        ]

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        enc_x, _ = self.lstm(x)
        enc_x = enc_x.view(enc_x.shape[0], 256, 4, 4)
        dec_x = self.decoder(enc_x)

        return Ae(dec_out=dec_x, enc_out=enc_x)


class AeV2V2V1MsdLstmV2(AeV2V2V1MsdLstmV1):

    def __init__(self,
                 chnum_in,
                 seed,
                 kl_loss_weight=1,
                 mse_loss_weight=1,
                 nu: float = 0.1,
                 lr=1e-4,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ LSTM + KL """
        super().__init__(
            chnum_in=chnum_in,
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
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")

    def training_step(self, train_batch, batch_idx):
        inputs, _, _ = train_batch
        self.center = self.center.to(inputs.device)
        outputs = self(inputs)
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        dist = torch.sum((enc_out - self.center)**2,
                         dim=tuple(range(1, enc_out.dim())))
        gaussian_target = F.softmax(torch.randn(enc_out.size(),
                                                device=enc_out.device),
                                    dim=1)
        kl_loss = self.kl_divergence(enc_out, gaussian_target)
        mse_loss = self.mse(inputs, dec_out)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = (svdd_loss + self.mse_loss_weight * mse_loss +
                self.kl_loss_weight * kl_loss)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}
