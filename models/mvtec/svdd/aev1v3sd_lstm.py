import torch
from utils import get_radius
import torch.nn as nn
import torch.nn.functional as F
from models.mvtec.base import MvTecSdV1
from models.base.sd import AeSd


class AeV1V3SdLstmV1(MvTecSdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[200],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """vanilla DALSVDD
        """
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
        self.rep_dim = rep_dim
        self.hidden_layer = self.rep_dim

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.lstm = nn.LSTM(self.rep_dim * 4 * 4,
                            self.hidden_layer * 4 * 4,
                            batch_first=True)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

        for name, param in self.lstm.named_parameters():

            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        lstm_x, (_, _) = self.lstm(x)
        # which do harm to performance of the model
        # x = self.bn2d(lstm_x)
        enc_x = self.fc1(lstm_x)

        return AeSd(enc_out=enc_x)
