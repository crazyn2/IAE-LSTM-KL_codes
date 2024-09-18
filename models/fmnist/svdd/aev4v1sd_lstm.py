from utils import get_radius
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.sd import AeSd, BaseSd


class AeV4V1SdLstmV1(BaseSd):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """Summary of class here.

        Longer class information....
        Longer class information....

        Attributes:
            rep_dim: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.hidden_layer = self.rep_dim

        # 考虑下修改神经网络初始化方式
        # https://zhuanlan.zhihu.com/p/405752148
        # LSTM输入输出大小是否会影响结果
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False, padding=2)
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
                            num_layers=num_layers,
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


class AeV4V1SdLstmV2(AeV4V1SdLstmV1):

    def __init__(self,
                 seed,
                 center=None,
                 kl_loss_weight=0.1,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ add latent gaussian constraint
        """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         num_layers=num_layers,
                         visual=visual,
                         objective=objective)
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")

    def training_step(self, train_batch, batch_idx):
        # self.current_training_step
        inputs, _ = train_batch
        outputs = self(inputs)
        enc_out = outputs.enc_out
        gaussian_target = F.softmax(torch.randn(enc_out.size(),
                                                device=enc_out.device),
                                    dim=1)
        kl_loss = self.kl_divergence(enc_out, gaussian_target)
        dist = torch.sum((enc_out - self.center)**2, dim=1)
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
        if self.visual:
            self.training_step_outputs.append(dist)
        loss = svdd_loss + self.kl_loss_weight * kl_loss
        self.log("train_loss", loss, sync_dist=True)
        # self.log("center_l2", (self.center**2).sum(), sync_dist=True)
        return {"loss": loss}


class AeV4V1SdLstmV3(AeV4V1SdLstmV2):

    def __init__(self,
                 seed,
                 center=None,
                 kl_loss_weight=0.1,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ add latent gaussian constraint
        """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         kl_loss_weight=kl_loss_weight,
                         rep_dim=rep_dim,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      amsgrad=self.optimizer_name == 'amsgrad')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestone, gamma=0.1)
        # return optimizer, scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class AeV4V1SdLstmV5(AeV4V1SdLstmV1):

    def __init__(self,
                 seed,
                 center=None,
                 num_layers=2,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.lstm = nn.LSTM(self.rep_dim * 4 * 2,
                            self.rep_dim * 4 * 4,
                            batch_first=True,
                            num_layers=num_layers)

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
        x = x.view(x.size(0), 2, 2 * 4 * 128)
        # x = torch.randn((100, 512, 4), device=x.device)
        lstm_x, (_, _) = self.lstm(x)
        # which do harm to performance of the model
        # x = self.bn2d(lstm_x)
        enc_x = self.fc1(lstm_x)

        return AeSd(enc_out=enc_x)
