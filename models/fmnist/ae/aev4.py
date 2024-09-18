import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_fabric.utilities.seed import seed_everything
from torchmetrics import AUROC
from models.base.ae import BaseAe
from sklearn.metrics import roc_auc_score
from models.base.ae import Ae


# lukasruff-Deep-SVDD-PyTorch/src/networks/cifar10_LeNet.py
class AeV4V1(BaseAe):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        super().__init__(
            seed=seed,
            rep_dim=rep_dim,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
        )

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

        return Ae(enc_out=enc_x, dec_out=dec_x)


class AeV4V2(AeV4V1):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        """ no final sigmoid layer """
        super().__init__(seed, rep_dim, lr, weight_decay, lr_milestones,
                         optimizer_name, visual)

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
        dec_x = self.deconv4(x)

        return Ae(enc_out=enc_x, dec_out=dec_x)


class AeV4V3(AeV4V1):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        log_red=False,
    ):
        super().__init__(seed, rep_dim, lr, weight_decay, lr_milestones,
                         optimizer_name, log_red)

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self(inputs +
                       torch.randn(inputs.size(), device=inputs.device))
        dec_out = outputs.dec_out
        mse_loss = self.mse(inputs, dec_out)
        if self.visual:
            mse_loss_scores = torch.sum((dec_out - inputs)**2,
                                        dim=tuple(range(1, dec_out.dim())))
            self.training_step_outputs.append(mse_loss_scores)
        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)
        self.log("train_loss", mse_loss)
        # self.log_tsne(outputs["dec_out"], self.current_epoch)
        return {'loss': mse_loss}


class AeV4V4(AeV4V1):

    def __init__(
        self,
        seed,
        rep_dim=128,
        kld_weight=1,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        log_red=False,
    ) -> None:
        super().__init__(seed, rep_dim, lr, weight_decay, lr_milestones,
                         optimizer_name, log_red)
        self.kld_weight = kld_weight
        self.fc_mu = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.fc_var = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        self.bce = nn.BCELoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        # mu = self.bn1d(mu)
        log_var = self.fc_var(x)
        # log_var = self.bn1d(log_var)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        enc_x = eps * std + mu
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

        return mu, log_var, dec_x

    def training_step(self, train_batch, batch_idx):
        inputs, _ = train_batch
        mu, log_var, recons = self(inputs)
        recons_loss = self.bce(recons, inputs)
        # recons_loss = self.mse(recons, inputs)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        self.log("train_loss", loss)
        # self.log_tsne(outputs["dec_out"], self.current_epoch)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        _, _, dec_out = self(inputs)
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            inputs.size(0), -1).sum(dim=1, keepdim=False)

        zip_params = [labels, mse_scores, l1_mse_scores]
        self.validation_step_outputs += list(zip(*zip_params))


class AeV4V5(AeV4V1):

    def __init__(
        self,
        seed,
        rep_dim=128,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        super().__init__(
            seed=seed,
            rep_dim=rep_dim,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
        )

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self(inputs)
        dec_out = outputs.dec_out
        enc_out = outputs.enc_out
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))

        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            inputs.size(0), -1).sum(dim=1, keepdim=False)

        zip_params = [labels, mse_scores, l1_mse_scores]
        if self.visual:
            # add additional record values
            zip_params += [enc_out, inputs]
        self.validation_step_outputs += list(zip(*zip_params))

    def on_validation_epoch_end(self):
        # torchmetrics
        # auroc = AUROC(task="binary")
        unpack_labels_scores = list(zip(*self.validation_step_outputs))
        labels = torch.stack(unpack_labels_scores[0])
        labels_np = labels.cpu().data.numpy()
        for i in range(0, len(self.aucroc_keys)):
            scores = torch.stack(unpack_labels_scores[i + 1])
            scores_np = scores.cpu().data.numpy()
            auroc_score_sk = roc_auc_score(labels_np, scores_np)
            self.log(self.aucroc_keys[i] + '_roc_auc_sk',
                     auroc_score_sk,
                     prog_bar=True,
                     sync_dist=True)

        if self.visual:
            visual_dim = torch.stack(unpack_labels_scores[-2])
            label_img = torch.stack(unpack_labels_scores[-1])
            metadata = labels.cpu().data.numpy()
            self.logger.experiment.add_embedding(
                visual_dim,
                metadata=metadata,
                label_img=label_img,
                global_step=self.global_step,
                # global_step=self.current_epoch,
            )
        self.validation_step_outputs.clear()
