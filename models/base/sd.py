import torch
import torch.nn.functional as F
from utils import get_radius
# from torchmetrics import AUROC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# import pytorch_lightning as pl
import lightning as pl
import numpy as np
from collections import namedtuple

AeSd = namedtuple("Ae", ["enc_out"])
AeMemSd = namedtuple("Ae", ["enc_out", "att"])


class BaseSd(pl.LightningModule):

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
        super().__init__()
        # super().__init__(seed)
        pl.seed_everything(seed, workers=True)
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_milestone = lr_milestones
        self.optimizer_name = optimizer_name
        self.visual = visual
        self.rep_dim = rep_dim
        self.center = torch.zeros(rep_dim) if center is None else center
        self.objective = objective
        self.warm_up_n_epochs = 10
        self.R = torch.tensor(0)
        self.nu = nu
        self.training_step_outputs = []
        self.validation_step_outputs = []

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
                outputs = net(inputs)
                if isinstance(outputs, dict):
                    enc_out = outputs["enc_out"]
                else:
                    enc_out = outputs.enc_out
                # enc_out = enc_out.contiguous().view(enc_out.size(0), -1)
                centers.append(enc_out)
                # n_samples += enc_out.shape[0]
                # c += torch.sum(enc_out, dim=0)
        c = torch.mean((torch.cat(centers)), dim=0).cuda()
        net.train()
        # c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def training_step(self, train_batch, batch_idx):
        # self.current_training_step
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs)
        if isinstance(outputs, dict):
            enc_out = outputs["enc_out"]
        else:
            enc_out = outputs.enc_out
            if self.global_step == 0:
                self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            # dist = self.R**2 + (1 / self.nu) * torch.max(
            #     torch.zeros_like(scores), scores)
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

    # def on_train_epoch_end(self) -> None:

    #     if self.log_red:
    #         training_scores = torch.cat(self.training_step_outputs)
    #         self.logger.experiment.add_histogram(
    #             tag='training_scores',
    #             values=training_scores,
    #             global_step=self.current_epoch)
    #         fig, ax = plt.subplots()
    #         ax.set_title('train_loss_scores')
    #         ax.hist(training_scores.cpu().data.numpy(),
    #                 alpha=0.5,
    #                 label='train',
    #                 bins='auto')
    #         ax.legend()
    #         self.logger.experiment.add_figure(tag='training_scores_hist',
    #                                           figure=fig,
    #                                           global_step=self.current_epoch)
    #     self.training_step_outputs.clear()

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)
        if isinstance(outputs, dict):
            enc_out = outputs["enc_out"]
        else:
            enc_out = outputs.enc_out

        dist = torch.sum((enc_out - self.center)**2,
                         dim=tuple(range(1, enc_out.dim())))
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        l1_svdd_scores = enc_out.sub(self.center).abs().contiguous().view(
            enc_out.size(0), -1).sum(dim=1, keepdim=False)
        # Save triples of (idx, label, score) in a list
        zip_params = [labels, svdd_scores, l1_svdd_scores]
        if self.visual:
            # add additional record values
            zip_params += [enc_out, inputs]
        self.validation_step_outputs += list(zip(*zip_params))

    def visual_hist(self, data: tuple, labels: tuple, title):
        fig, ax = plt.subplots()
        ax.set_title(title + '_hist')
        cat_data = np.concatenate(data)
        ax.hist(
            data,
            alpha=0.5,
            range=[cat_data.min(), np.percentile(cat_data, 99)],
            label=labels,
            # density=True,
            bins='auto')
        ax.legend()
        self.logger.experiment.add_figure(tag=title + '_hist',
                                          figure=fig,
                                          global_step=self.current_epoch)

    def on_validation_epoch_end(self):
        # torchmetrics
        # auroc = AUROC(task="binary")
        aucroc_keys = ['svdd', 'l1_svdd']
        unpack_labels_scores = list(zip(*self.validation_step_outputs))
        labels = torch.stack(unpack_labels_scores[0])
        labels_np = labels.cpu().data.numpy()
        for i in range(0, len(aucroc_keys)):
            scores = torch.stack(unpack_labels_scores[i + 1])
            # torchmetrics aucroc
            # auroc_score_trh = auroc(scores, labels)
            # sklearn.metrics aucroc
            scores_np = scores.cpu().data.numpy()
            auroc_score_sk = roc_auc_score(labels_np, scores_np)
            # self.log(aucroc_keys[i] + '_roc_auc_trh',
            #          auroc_score_trh,
            #          prog_bar=True,
            #          sync_dist=True)
            self.log(aucroc_keys[i] + '_roc_auc_sk',
                     auroc_score_sk,
                     prog_bar=True,
                     sync_dist=True)
            if self.visual:
                self.logger.experiment.add_histogram(
                    tag=aucroc_keys[i] + '_scores',
                    values=torch.stack(unpack_labels_scores[1]),
                    global_step=self.current_epoch)
                scores_np_normal = scores_np[labels_np == 0]
                scores_np_abnormal = scores_np[labels_np == 1]
                self.visual_hist([scores_np_normal, scores_np_abnormal],
                                 ['0', '1'], aucroc_keys[i])

        if self.visual:
            visual_dim = torch.stack(unpack_labels_scores[-2])
            visual_dim = torch.cat((visual_dim, self.center.unsqueeze(0)))
            label_img = torch.stack(unpack_labels_scores[-1])
            label_img = torch.cat((label_img, torch.rand(
                label_img[0].size()).unsqueeze(0).to(label_img.device)))
            metadata = torch.cat(
                (labels, torch.tensor(2).unsqueeze(0).to(labels.device)))
            self.logger.experiment.add_embedding(
                visual_dim,
                metadata=metadata.cpu().data.numpy(),
                label_img=label_img,
                tag=self.objective,
                global_step=self.global_step,
                # global_step=self.current_epoch,
            )
        #     # only when not empty
        #     if len(self.training_step_outputs) != 0:
        #         svdd_scores_np = torch.stack(
        #             unpack_labels_scores[1]).cpu().data.numpy()
        #         training_scores_np = torch.cat(
        #             self.training_step_outputs).cpu().data.numpy()
        #         self.logger.experiment.add_histogram(
        #             tag='training_scores',
        #             values=training_scores_np,
        #             global_step=self.current_epoch)
        #         self.visual_hist(
        #             [training_scores_np, svdd_scores_np[labels_np == 0]],
        #             ['train', 'test'], 'train_test_scores')
        #         self.training_step_outputs.clear()
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
