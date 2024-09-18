import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC
import torch
from utils import get_radius
import pytorch_lightning as pl
from losses import EntropyLossEncap
from models.base.sd import BaseSd


class AeV2V1SdV1(BaseSd):
    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestone=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestone,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.chnum_in = 1
        self.feature_num = 128,
        self.feature_num_2 = 96,
        self.feature_num_x2 = 256,
        self.lstm = nn.LSTM(self.rep_dim, self.rep_dim, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in,
                      self.feature_num_2,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(self.feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_num_2,
                      self.feature_num,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False), nn.BatchNorm2d(self.feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_num,
                      self.feature_num_x2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False), nn.BatchNorm2d(self.feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_num_x2,
                      self.feature_num_x2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False), nn.BatchNorm2d(self.feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_num_x2,
                               self.feature_num_x2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feature_num_x2), nn.LeakyReLU(0.2,
                                                              inplace=True),
            nn.ConvTranspose2d(self.feature_num_x2,
                               self.feature_num,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=False), nn.BatchNorm2d(self.feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_num,
                               self.feature_num_2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=False), nn.BatchNorm2d(self.feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_num_2,
                               self.chnum_in,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=False))
        for name, param in self.lstm.named_parameters():
            # nn.init.uniform_(param, -0.1, 0.1)
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain('leaky_relu'))
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        # encoder
        enc_x = self.encoder(x)
        return {"enc_out": enc_x}
