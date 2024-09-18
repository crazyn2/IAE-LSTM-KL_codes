import torch.nn as nn
import torch.nn.functional as F
from .aev1sd import AEV1SdV1


class AEV1SdGruV1(AEV1SdV1):
    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=16,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestone=[50],
                 optimizer_name='amsgrad',
                 log_red=False,
                 objective='one-class'):
        super().__init__(seed, center, nu, rep_dim, lr, weight_decay,
                         lr_milestone, optimizer_name, log_red, objective)
        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256, bias=False)
        self.enc2 = nn.Linear(in_features=256, out_features=128, bias=False)
        self.enc3 = nn.Linear(in_features=128, out_features=64, bias=False)
        # self.enc4 = nn.Linear(in_features=64, out_features=32, bias=False)
        self.gru = nn.GRU(64, 32, batch_first=True)
        for name, param in self.lstm.named_parameters():
            # nn.init.uniform_(param, -0.1, 0.1)
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        self.enc5 = nn.Linear(in_features=32,
                              out_features=self.rep_dim,
                              bias=False)
        for name, param in self.gru.named_parameters():
            # nn.init.uniform_(param, -0.1, 0.1)
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        # decoder
        # self.dec1 = nn.Linear(in_features=16, out_features=32)
        # self.dec2 = nn.Linear(in_features=32, out_features=64)
        # self.dec3 = nn.Linear(in_features=64, out_features=128)
        # self.dec4 = nn.Linear(in_features=128, out_features=256)
        # self.dec5 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        # x_input = x
        x = x.view(x.size(0), -1)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        # x = F.relu(self.enc4(x))
        lstm, (_, _) = self.lstm(x)
        # x = F.relu(lstm)
        enc_x = F.relu(self.enc5(lstm))
        # x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        # x = x.view(x_input.shape)
        return {"enc_out": enc_x}


class AEV1SdGruV2(AEV1SdV1):
    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=16,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestone=50,
                 optimizer_name='amsgrad',
                 log_red=False,
                 objective='one-class'):
        super().__init__(seed, center, nu, rep_dim, lr, weight_decay,
                         lr_milestone, optimizer_name, log_red, objective)
        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        # for name, param in self.lstm.named_parameters():
        #     # nn.init.uniform_(param, -0.1, 0.1)
        #     if name.startswith("weight"):
        #         nn.init.orthogonal_(param)
        #     else:
        #         nn.init.zeros_(param)
        self.enc5 = nn.Linear(in_features=32, out_features=self.rep_dim)
        # decoder
        # self.dec1 = nn.Linear(in_features=16, out_features=32)
        # self.dec2 = nn.Linear(in_features=32, out_features=64)
        # self.dec3 = nn.Linear(in_features=64, out_features=128)
        # self.dec4 = nn.Linear(in_features=128, out_features=256)
        # self.dec5 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        # x_input = x
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        x = F.leaky_relu(self.enc3(x))
        x = F.leaky_relu(self.enc4(x))
        lstm, (_, _) = self.lstm(x)
        x = F.leaky_relu(lstm)
        x = F.leaky_relu(self.enc5(lstm))
        # x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        # x = x.view(x_input.shape)
        return {"lnr_out": x}


class FMNISTLnrLstmV3(AEV1SdV1):
    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=16,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestone=[50],
                 optimizer_name='amsgrad',
                 log_red=False,
                 objective='one-class'):
        super().__init__(seed, center, nu, rep_dim, lr, weight_decay,
                         lr_milestone, optimizer_name, log_red, objective)
        # encoder
        # self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.lstm = nn.LSTM(784, 256, batch_first=True)
        # for name, param in self.lstm.named_parameters():
        #     # nn.init.uniform_(param, -0.1, 0.1)
        #     if name.startswith("weight"):
        #         nn.init.orthogonal_(param)
        #     else:
        #         nn.init.zeros_(param)
        self.enc5 = nn.Linear(in_features=32, out_features=self.rep_dim)
        # decoder
        # self.dec1 = nn.Linear(in_features=16, out_features=32)
        # self.dec2 = nn.Linear(in_features=32, out_features=64)
        # self.dec3 = nn.Linear(in_features=64, out_features=128)
        # self.dec4 = nn.Linear(in_features=128, out_features=256)
        # self.dec5 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        # x_input = x
        x = x.view(x.size(0), -1)
        lstm, (_, _) = self.lstm(x)
        lstm = F.relu(lstm)
        x = F.relu(self.enc2(lstm))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        # lstm, (_, _) = self.lstm(x)
        x = self.enc5(x)
        # x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        # x = x.view(x_input.shape)
        return {"lnr_out": x}
