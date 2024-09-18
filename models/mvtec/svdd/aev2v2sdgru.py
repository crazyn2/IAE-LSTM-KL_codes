from models.ped2.base import Ped2AeSdV1
import torch
from utils import get_radius
from torch import nn
# from torcheval.metrics import BinaryAUROC
from models.base.sd import AeSd
from models.ped2.ae import EncoderV2


class AeV2V2V1SdGruV1(Ped2AeSdV1):

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
        self.fc1 = nn.Linear(256 * 16 * 16, 256 * 4 * 4, bias=False)
        self.lstm = nn.GRU(256 * 4 * 4, 256 * 4 * 4, batch_first=True)
        # self.lstm = nn.GRU(256 * 16 * 16, 256 * 16 * 16, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        self.aucroc_keys = ['svdd', 'l1_svdd']

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        enc_x, _ = self.lstm(x)

        return AeSd(enc_out=enc_x)

    def training_step(self, train_batch, batch_idx):
        inputs, _, _ = train_batch
        self.center = self.center.to(inputs.device)
        outputs = self(inputs)
        enc_out = outputs.enc_out
        dist = torch.sum((enc_out - self.center)**2,
                         dim=tuple(range(1, enc_out.dim())))
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels, scenes = val_batch
        self.center = self.center.to(inputs.device)
        # print(scenes)
        labels = labels[:, labels.shape[1] // 2]
        scenes = scenes[:, scenes.shape[1] // 2]
        enc_out = self(inputs).enc_out
        dist = torch.sum((enc_out - self.center)**2,
                         dim=tuple(range(1, enc_out.dim())))
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        l1_svdd_scores = torch.sum((enc_out - self.center).abs(),
                                   dim=tuple(range(1, enc_out.dim())))
        # Save triples of (idx, label, score) in a list
        zip_params = [labels, svdd_scores, l1_svdd_scores, scenes]
        self.validation_step_outputs += list(zip(*zip_params))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        # return optimizer, scheduler
        return {"optimizer": optimizer}
