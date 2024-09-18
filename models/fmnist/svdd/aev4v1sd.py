import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import get_radius
from models.base.sd import BaseSd
from models.base.sd import AeSd


class AeV4V1SdV1(BaseSd):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ vanilla sd """
        super().__init__(seed, center, nu, rep_dim, lr, weight_decay,
                         lr_milestones, optimizer_name, visual, objective)
        self.pool = nn.MaxPool2d(2, 2)
        self.rep_dim = rep_dim
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
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        enc_x = self.fc1(x)

        return AeSd(enc_out=enc_x)


class AeV4V1SdV2(AeV4V1SdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ denoising center + svdd """
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

    def init_center_c(self, net, train_loader, eps=0.1):
        # n_samples = 0
        # c = torch.zeros(self.rep_dim).cuda()

        centers = []
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.cuda()
                outputs = net(inputs +
                              torch.randn(inputs.size(), device=inputs.device))
                enc_out = outputs.enc_out
                # enc_out = enc_out.contiguous().view(enc_out.size(0), -1)
                centers.append(enc_out)
                # n_samples += enc_out.shape[0]
                # c += torch.sum(enc_out, dim=0)
        c = torch.mean((torch.cat(centers)), dim=0).cuda()
        # c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    # def training_step(self, train_batch, batch_idx):
    #     # self.current_training_step
    #     inputs, labels = train_batch
    #     # self.decoder.eval()
    #     outputs = self(inputs)
    #     enc_out = outputs.enc_out
    #     if self.global_step == 0:
    #         self.logger.experiment.add_graph(self, inputs)
    #     dist = torch.sum((enc_out - self.center)**2, dim=1)
    #     if self.objective == 'soft-boundary':
    #         scores = dist - self.R**2
    #         dist = self.R**2 + (1 / self.nu) * torch.max(
    #             torch.zeros_like(scores), scores)
    #         loss = self.R**2 + (1 / self.nu) * torch.mean(
    #             torch.max(torch.zeros_like(scores), scores))
    #     else:
    #         loss = torch.mean(dist)
    #     if (self.objective == 'soft-boundary') and (self.current_epoch >=
    #                                                 self.warm_up_n_epochs):
    #         self.R.data = torch.tensor(get_radius(dist, self.nu),
    #                                    device=self.device)
    #     if self.visual:
    #         self.training_step_outputs.append(dist)

    #     # if self.global_step == 0:
    #     #     self.logger.experiment.add_graph(self, inputs)

    #     self.log("train_loss", loss, sync_dist=True)
    #     # self.log("center_l2", (self.center**2).sum(), sync_dist=True)
    #     return {"loss": loss}


class AeV4V1SdV3(AeV4V1SdV2):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ denoising center + svdd """
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

    def training_step(self, train_batch, batch_idx):
        # self.current_training_step
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs +
                       torch.randn(inputs.size(), device=inputs.device))
        enc_out = outputs.enc_out
        if self.global_step == 0:
            self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        # self.log("center_l2", (self.center**2).sum(), sync_dist=True)
        return {"loss": loss}


class AeV4V1SdV4(AeV4V1SdV1):

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
        """ kl gaussian """
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


class AeV4V1SdV5(AeV4V1SdV1):

    def __init__(self,
                 seed,
                 center=None,
                 kl_loss_weight=0.1,
                 mean=0,
                 std=1,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ kl gaussian """
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
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")
        self.mean = mean
        self.std = std

    def training_step(self, train_batch, batch_idx):
        # self.current_training_step
        inputs, _ = train_batch
        outputs = self(inputs)
        enc_out = outputs.enc_out
        gaussian_target = F.softmax(torch.normal(mean=self.mean,
                                                 std=self.std,
                                                 size=enc_out.size(),
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
