from torch import nn
import lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from utils import get_radius


class MvTecSdV1(pl.LightningModule):

    def __init__(self,
                 seed,
                 rep_dim=128,
                 nu: float = 0.1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__()
        self.save_hyperparameters()
        # print(self.hparams)
        pl.seed_everything(seed, workers=True)
        self.lr = lr
        self.rep_dim = rep_dim
        self.weight_decay = weight_decay
        self.lr_milestones = lr_milestones
        self.optimizer_name = optimizer_name
        self.objective = objective
        self.warm_up_n_epochs = 10
        self.R = torch.tensor(0)
        self.nu = nu
        self.mse = nn.MSELoss(reduction='mean')
        self.visual = visual
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.aucroc_keys = ['svdd', 'l1_svdd']

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
        net.train()
        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

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
        inputs, _, labels = val_batch
        self.center = self.center.to(inputs.device)
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
        zip_params = [labels, svdd_scores, l1_svdd_scores]
        self.validation_step_outputs += list(zip(*zip_params))

    def on_validation_epoch_end(self):
        # torchmetrics
        # auroc = AUROC(task="binary")
        # auroc = AUROC(task="binary", average="none")
        # metric = BinaryAUROC()
        unpacked_labels_scores = list(zip(*self.validation_step_outputs))
        labels = torch.stack(unpacked_labels_scores[0])
        # mse_scores = torch.stack(unpacked_labels_scores[1])
        labels_np = labels.cpu().data.numpy()

        for i in range(0, len(self.aucroc_keys)):
            scores = torch.stack(unpacked_labels_scores[i + 1])
            scores_np = scores.cpu().data.numpy()
            auroc_sk = roc_auc_score(labels_np, scores_np)
            self.log(self.aucroc_keys[i] + '_auc',
                     auroc_sk,
                     prog_bar=True,
                     sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.optimizer_name == 'amsgrad')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1)
        # return optimizer, scheduler
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
