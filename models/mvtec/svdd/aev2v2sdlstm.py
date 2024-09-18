from models.mvtec.base import MvTecSdV1
import torch
from torch import nn
from models.base.sd import AeSd
from models.ped2.ae import EncoderV2
from utils import get_radius
import torch.nn.functional as F


class AeV2V2V1SdLstmV1(MvTecSdV1):

    def __init__(self,
                 chnum_in,
                 seed,
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
        # self.fc1 = torch.nn.Linear(256 * 4 * 4, 256 * 4 * 4, bias=False)
        self.lstm = torch.nn.LSTM(256 * 4 * 4, 256 * 4 * 4, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        self.aucroc_keys = ['svdd', 'l1_svdd']

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        enc_x, _ = self.lstm(x)

        return AeSd(enc_out=enc_x)


class AeV2V2V1SdLstmV2(AeV2V2V1SdLstmV1):

    def __init__(self,
                 seed,
                 chnum_in=3,
                 kl_loss_weight=1,
                 nu: float = 0.1,
                 lr=1e-4,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ LSTM+KL """
        super().__init__(
            seed=seed,
            chnum_in=chnum_in,
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

    def training_step(self, train_batch, batch_idx):
        inputs, _, _ = train_batch
        self.center = self.center.to(inputs.device)
        outputs = self(inputs)
        enc_out = outputs.enc_out
        dist = torch.sum((enc_out - self.center)**2,
                         dim=tuple(range(1, enc_out.dim())))
        gaussian_target = F.softmax(torch.randn(enc_out.size(),
                                                device=enc_out.device),
                                    dim=1)
        kl_loss = self.kl_divergence(enc_out, gaussian_target)
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
        loss = (svdd_loss + self.kl_loss_weight * kl_loss)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}


class AeV2V2V1SdLstmV3(AeV2V2V1SdLstmV2):

    def __init__(self,
                 seed,
                 chnum_in=3,
                 kl_loss_weight=1,
                 nu: float = 0.1,
                 lr=1e-4,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ LSTM+KL """
        super().__init__(
            seed=seed,
            chnum_in=chnum_in,
            kl_loss_weight=kl_loss_weight,
            nu=nu,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
            objective=objective,
        )
        self.hidden_size = 256 * 4 * 4
        self.lstm = nn.LSTMCell(256 * 4 * 4, self.hidden_size)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        h_0 = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        c_0 = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        # x = self.fc1(x)
        enc_x, _ = self.lstm(x, (h_0, c_0))
        return AeSd(enc_out=enc_x)
