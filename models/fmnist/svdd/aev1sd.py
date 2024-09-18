import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC
import torch
from utils import get_radius
import pytorch_lightning as pl
from losses import EntropyLossEncap
from lightning.fabric import seed_everything


class AEV1SdV1(pl.LightningModule):
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
        super().__init__()
        seed_everything(seed)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestone = lr_milestone
        self.optimizer_name = optimizer_name
        self.log_red = log_red
        self.save_hyperparameters()
        self.rep_dim = rep_dim
        self.center = torch.zeros(rep_dim) if center is None else center
        self.objective = objective
        self.warm_up_n_epochs = 10
        self.R = torch.tensor(0)
        self.nu = nu
        self.validation_step_outputs = []
        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256, bias=False)
        self.enc2 = nn.Linear(in_features=256, out_features=128, bias=False)
        self.enc3 = nn.Linear(in_features=128, out_features=64, bias=False)
        self.enc4 = nn.Linear(in_features=64, out_features=32, bias=False)
        self.enc5 = nn.Linear(in_features=32,
                              out_features=self.rep_dim,
                              bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        enc_x = self.enc5(x)
        return {"enc_out": enc_x}

    def init_center_c(self, net, train_loader, eps=0.1):
        n_samples = 0
        c = torch.zeros(self.rep_dim).cuda()
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.cuda()
                outputs = net(inputs)
                enc_out = outputs["enc_out"]
                n_samples += enc_out.shape[0]
                c += torch.sum(enc_out, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs)
        enc_out = outputs["enc_out"]
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch >=
                                                    self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)
        enc_out = outputs['enc_out']
        # dec_out = outputs["dec_out"]
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
        else:
            scores = dist
        self.validation_step_outputs += zip(labels, scores)

    def on_validation_epoch_end(self):
        auroc = AUROC(task="binary")
        labels, scores = zip(*self.validation_step_outputs)
        auroc_score = auroc(torch.stack(scores), torch.stack(labels))
        self.log('val_roc_auc', auroc_score, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.optimizer_name == 'amsgrad')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestone, gamma=0.1)
        # return optimizer, scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class AEV1SdV2(AEV1SdV1):
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
        self.enc4 = nn.Linear(in_features=64, out_features=32, bias=False)
        self.enc5 = nn.Linear(in_features=32,
                              out_features=self.rep_dim,
                              bias=False)
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
        x = self.enc5(x)
        # x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        # x = x.view(x_input.shape)
        return {"enc_out": x}
