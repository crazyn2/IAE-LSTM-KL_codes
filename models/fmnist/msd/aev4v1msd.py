import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import get_radius
from models.base.msd import BaseMsdV1
from models.base.ae import Ae
from sklearn.metrics import roc_auc_score
import os


class AeV4V1MsdV1(BaseMsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed, center, nu, rep_dim, mse_loss_weight, lr,
                         weight_decay, lr_milestones, optimizer_name, visual,
                         objective)
        # BaseSd.__init__(self, seed, center, nu, lr, weight_decay, lr_milestone,
        #                 optimizer_name, log_red, objective)
        # AeV4V1.__init__(self, seed, rep_dim, lr, weight_decay, lr_milestone,
        #                 optimizer_name, log_red)
        # Encoder (must match the Deep SVDD network above)
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
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)),
                                          128,
                                          5,
                                          bias=False,
                                          padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        enc_x = self.fc1(x)
        x = self.bn1d(enc_x)
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        dec_x = torch.sigmoid(x)

        return Ae(dec_out=dec_x, enc_out=enc_x)


class AeV4V1MsdV2(AeV4V1MsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ aemsd """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
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
                outputs = net(inputs)
                dec_out = outputs.dec_out
                # enc_out = enc_out.contiguous().view(enc_out.size(0), -1)
                centers.append(dec_out)
                # n_samples += enc_out.shape[0]
                # c += torch.sum(enc_out, dim=0)
        c = torch.mean((torch.cat(centers)), dim=0).cuda()
        # c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs)
        dec_out = outputs.dec_out
        if self.global_step == 0:
            self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((dec_out - self.center)**2,
                         dim=tuple(range(1, dec_out.dim())))
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = svdd_loss
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)
        dec_out = outputs.dec_out
        dist = torch.sum((dec_out - self.center)**2,
                         dim=tuple(range(1, dec_out.dim())))
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        # l1 score
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            dec_out.size(0), -1).sum(dim=1, keepdim=False)
        l1_svdd_scores = dec_out.sub(self.center).abs().contiguous().view(
            dec_out.size(0), -1).sum(dim=1, keepdim=False)
        svdd_mse_scores = svdd_scores + self.mse_loss_weight * mse_scores
        # Save triples of (idx, label, score) in a list
        zip_params = [
            labels,
            svdd_scores,
            mse_scores,
            svdd_mse_scores,
            l1_mse_scores,
            l1_svdd_scores,
        ]
        if self.visual:
            # add additional record values
            zip_params += [dec_out, inputs]
        self.validation_step_outputs += list(zip(*zip_params))


class AeV4V1MsdV3(AeV4V1MsdV2):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ aemsd + msd"""
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs)
        dec_out = outputs.dec_out
        if self.global_step == 0:
            self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((dec_out - self.center)**2,
                         dim=tuple(range(1, dec_out.dim())))
        mse_loss = self.mse(inputs, dec_out)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = svdd_loss + self.mse_loss_weight * mse_loss
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}


class AeV4V1MsdV4(AeV4V1MsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """ kl guassian """
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs)
        if isinstance(outputs, dict):
            enc_out = outputs["enc_out"]
            dec_out = outputs["dec_out"]
        else:
            enc_out = outputs.enc_out
            dec_out = outputs.dec_out
            if self.global_step == 0:
                self.logger.experiment.add_graph(self, inputs)
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        gaussian_target = F.softmax(torch.randn(enc_out.size(),
                                                device=enc_out.device),
                                    dim=1)
        kl_loss = self.kl_divergence(enc_out, gaussian_target)
        mse_loss = self.mse(inputs, dec_out)
        if self.objective == 'soft-boundary':
            scores = dist - self.R**2
            dist = self.R**2 + (1 / self.nu) * torch.max(
                torch.zeros_like(scores), scores)
            svdd_loss = self.R**2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores))
        else:
            svdd_loss = torch.mean(dist)
        if (self.objective == 'soft-boundary') and (self.current_epoch
                                                    >= self.warm_up_n_epochs):
            self.R.data = torch.tensor(get_radius(dist, self.nu),
                                       device=self.device)
        loss = (svdd_loss + self.mse_loss_weight * mse_loss +
                self.kl_loss_weight * kl_loss)
        if self.visual:
            self.training_step_outputs.append(dist)

        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        # torchmetrics
        # auroc = AUROC(task="binary")
        unpack_labels_scores = list(zip(*self.validation_step_outputs))
        labels = torch.stack(unpack_labels_scores[0])
        labels_np = labels.cpu().data.numpy()
        torch.save(
            unpack_labels_scores,
            os.path.join(
                self.trainer.log_dir,
                "labels_scores-epoch=%03d.pt" % self.current_epoch))
        for i in range(0, len(self.aucroc_keys)):
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
            self.log(self.aucroc_keys[i] + '_roc_auc_sk',
                     auroc_score_sk,
                     prog_bar=True,
                     sync_dist=True)
            if self.visual:
                self.logger.experiment.add_histogram(
                    tag=self.aucroc_keys[i] + '_scores',
                    values=torch.stack(unpack_labels_scores[1]),
                    global_step=self.current_epoch)
                scores_np_normal = scores_np[labels_np == 0]
                scores_np_abnormal = scores_np[labels_np == 1]
                self.visual_hist([scores_np_normal, scores_np_abnormal],
                                 ['0', '1'], self.aucroc_keys[i])

        if self.visual:
            visual_dim = torch.stack(unpack_labels_scores[-2])
            visual_dim = torch.cat((visual_dim, self.center.unsqueeze(0)))
            label_img = torch.stack(unpack_labels_scores[-1])
            label_img = torch.cat((label_img, torch.rand(
                label_img[0].size()).unsqueeze(0).to(label_img.device)))
            metadata = torch.cat(
                (labels, torch.tensor(2).unsqueeze(0).to(labels.device)))
            metadata = metadata.cpu().data.numpy()
            self.logger.experiment.add_embedding(
                visual_dim,
                metadata=metadata,
                label_img=label_img,
                tag=self.objective,
                global_step=self.global_step,
                # global_step=self.current_epoch,
            )
            # only when not empty
            if len(self.training_step_outputs) != 0:
                svdd_scores_np = torch.stack(
                    unpack_labels_scores[1]).cpu().data.numpy()
                training_scores_np = torch.cat(
                    self.training_step_outputs).cpu().data.numpy()
                self.logger.experiment.add_histogram(
                    tag='training_scores',
                    values=training_scores_np,
                    global_step=self.current_epoch)
                self.visual_hist(
                    [training_scores_np, svdd_scores_np[labels_np == 0]],
                    ['train', 'test'], 'train_test_scores')
                self.training_step_outputs.clear()

        self.validation_step_outputs.clear()
