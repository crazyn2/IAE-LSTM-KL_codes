from torch import nn
import lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from models.base.ae import Ae


class MvTecAeV2(pl.LightningModule):

    def __init__(
        self,
        seed,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        # print(self.hparams)
        pl.seed_everything(seed, workers=True)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestones = lr_milestones
        self.optimizer_name = optimizer_name
        self.mse = nn.MSELoss(reduction='mean')
        self.visual = visual
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.aucroc_keys = ['mse', 'l1_mse', 'neg_psnr']

    def forward(self, x):
        enc_x = self.encoder(x)
        x = enc_x.repeat(1, 2, 1, 1)
        dec_x = self.decoder(x)
        return Ae(dec_out=dec_x, enc_out=enc_x)

    def training_step(self, train_batch, batch_idx):
        inputs, _, _ = train_batch
        outputs = self(inputs)
        dec_out = outputs.dec_out
        mse_loss = self.mse(inputs, dec_out)
        loss = mse_loss
        self.log("train_loss", loss)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, _, labels = val_batch
        dec_out = self(inputs).dec_out
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            inputs.size(0), -1).sum(dim=1, keepdim=False)
        psnr_scores = 10 * torch.log10(1 / mse_scores)
        neg_psnr_scores = -psnr_scores
        # neg_psnr_scores = -psnr_scores
        # labels = labels.squeeze(1)
        zip_params = [
            labels, mse_scores, l1_mse_scores, neg_psnr_scores, psnr_scores
        ]
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
