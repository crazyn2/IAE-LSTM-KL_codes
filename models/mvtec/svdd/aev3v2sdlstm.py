from torch import nn
import torch
from utils import get_radius
# from torcheval.metrics import BinaryAUROC
from models.base.sd import AeSd
from sklearn.metrics import roc_auc_score
from models.base.sd import BaseSd


class AeV3V2V1SdLstmV1(BaseSd):

    def __init__(self,
                 chnum_in,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 lr=1e-4,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(
            seed=seed,
            center=center,
            nu=nu,
            rep_dim=rep_dim,
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
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in,
                      feature_num_2, (3, 3, 3),
                      stride=(1, 2, 2),
                      padding=(1, 1, 1)), nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2,
                      feature_num, (3, 3, 3),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1)), nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num,
                      feature_num_x2, (3, 3, 3),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1)), nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2,
                      feature_num_x2, (3, 3, 3),
                      stride=(2, 2, 2),
                      padding=(1, 1, 1)), nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc1 = nn.Linear(8192, 256 * 4, bias=False)
        self.lstm = nn.LSTM(256 * 4, 256 * 4, batch_first=True)
        self.aucroc_keys = ['svdd', 'l1_svdd']

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        enc_x, (_, _) = self.lstm(x)
        return AeSd(enc_out=enc_x)

    def init_center_c(self, net, train_loader, eps=0.1):

        centers = []
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for train_batch in train_loader:
                # get the inputs of the batch
                inputs, _, _ = train_batch
                inputs = inputs.cuda()
                outputs = net(inputs)
                enc_out = outputs.enc_out
                centers.append(enc_out)
        c = torch.mean((torch.cat(centers)), dim=0).cuda()

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def training_step(self, train_batch, batch_idx):
        inputs, _, _ = train_batch
        outputs = self(inputs)
        enc_out = outputs.enc_out
        dist = torch.mean((enc_out - self.center)**2, dim=1)
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
        # print(scenes)
        labels = labels[:, labels.shape[1] // 2]
        scenes = scenes[:, scenes.shape[1] // 2]
        enc_out = self(inputs).enc_out
        dist = torch.mean((enc_out - self.center)**2,
                          dim=tuple(range(1, enc_out.dim())))
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        l1_svdd_scores = torch.mean((enc_out - self.center).abs(),
                                    dim=tuple(range(1, enc_out.dim())))
        # Save triples of (idx, label, score) in a list
        zip_params = [labels, svdd_scores, l1_svdd_scores, scenes]
        self.validation_step_outputs += list(zip(*zip_params))

    def min_max(self, scores, mask):
        psnr_vec = scores[mask]
        psnr_max = torch.max(psnr_vec)
        psnr_min = torch.min(psnr_vec)
        normalized_psnr = (psnr_vec - psnr_min) / (psnr_max - psnr_min)
        scores[mask] = normalized_psnr
        return scores

    def on_validation_epoch_end(self):
        # torchmetrics
        unpacked_labels_scores = list(zip(*self.validation_step_outputs))
        labels = torch.stack(unpacked_labels_scores[0])
        labels_np = labels.cpu().data.numpy()
        scenes = torch.stack(unpacked_labels_scores[-1])
        uni_scene = torch.unique(scenes)
        for i in range(0, len(self.aucroc_keys)):
            scores = torch.stack(unpacked_labels_scores[i + 1])
            for scene_index in range(len(uni_scene)):
                mask = (scenes == uni_scene[scene_index])
                self.min_max(scores, mask)
            scores_np = scores.cpu().data.numpy()
            auroc_sk = roc_auc_score(labels_np, scores_np)
            self.log(self.aucroc_keys[i] + '_auc',
                     auroc_sk,
                     prog_bar=True,
                     sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        # return optimizer, scheduler
        return {"optimizer": optimizer}
