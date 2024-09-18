import torch
import math
from utils import get_radius
import torch.nn as nn
import torch.nn.functional as F
from models.base.msd import BaseMsdV1
from models.base.ae import Ae
import numpy as np
from sklearn.metrics import roc_auc_score
from models.base.lstm import MyLSTMV0
from models.base.lstm import MyLSTMV1
from models.base.lstm import MyLSTMV2
from models.base.lstm import MyLSTMV3
from models.base.lstm import MyLSTMV4
from models.base.lstm import MyLSTMV5
from models.base.lstm import MyLSTMV6
from models.base.lstm import MyLSTMV7


class AeV4V1MsdLstmV1(BaseMsdV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 weight_decay=0.5e-6,
                 num_layers=1,
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
        super().__init__(seed, center, nu, rep_dim, mse_loss_weight, lr,
                         weight_decay, lr_milestones, optimizer_name, visual,
                         objective)
        self.hidden_layer = rep_dim
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
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.bn2d = nn.BatchNorm1d(self.rep_dim * 4 * 4,
                                   eps=1e-04,
                                   affine=False)

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
        for name, param in self.lstm.named_parameters():
            # nn.init.uniform_(param, -0.1, 0.1)
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


class AeV4V1MsdLstmV2(AeV4V1MsdLstmV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """add kl divergence with gaussian noise

        Attributes:
            rep_dim: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """
        # super().__init__(seed, center, nu, rep_dim, mse_loss_weight, lr,
        #                  weight_decay, lr_milestone, optimizer_name, visual,
        #                  objective)
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
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


class AeV4V1MsdLstmV3(AeV4V1MsdLstmV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """add kl divergence with gaussian noise

        Attributes:
            rep_dim: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """
        # super().__init__(seed, center, nu, rep_dim, mse_loss_weight, lr,
        #                  weight_decay, lr_milestone, optimizer_name, visual,
        #                  objective)
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")
        self.lstm = MyLSTMV1(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        lstm_x, gates = self.lstm(x)
        # which do harm to performance of the model
        # x = self.bn2d(lstm_x)
        enc_x = self.fc1(lstm_x)
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

        return Ae(dec_out=dec_x, enc_out=enc_x), gates

    def init_center_c(self, net, train_loader, eps=0.1):
        """ init center vector of deep SVDD """
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
                outputs, _ = net(inputs)
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

    def training_step(self, train_batch, batch_idx):
        inputs, _ = train_batch
        # self.decoder.eval()
        outputs, _ = self(inputs)
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)
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

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs, gates = self(inputs)
        mse_gates = []
        for i in range(len(gates)):
            mse_gates.append(
                torch.sum(gates[i], dim=tuple(range(1, gates[i].dim()))))
        h_t, c_t, c_tn, h_tn, h_tnn, h_tonn = mse_gates
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        # l1 score
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            dec_out.size(0), -1).sum(dim=1, keepdim=False)
        l1_svdd_scores = enc_out.sub(self.center).abs().contiguous().view(
            enc_out.size(0), -1).sum(dim=1, keepdim=False)
        svdd_mse_scores = (svdd_scores - svdd_scores.min()) / (svdd_scores.max(
        ) - svdd_scores.min()) + (mse_scores - mse_scores.min()) / (
            mse_scores.max() - mse_scores.min())
        # Save triples of (idx, label, score) in a list
        zip_params = [
            labels, svdd_scores, mse_scores, svdd_mse_scores, l1_mse_scores,
            l1_svdd_scores, h_t, c_t, c_tn, h_tn, h_tnn, h_tonn
        ]
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
                scores_np_normal = scores_np[labels_np == 0]
                scores_np_abnormal = scores_np[labels_np == 1]
                self.log(self.aucroc_keys[i] + '_sc0',
                         np.mean(scores_np_normal),
                         prog_bar=True,
                         sync_dist=True)
                self.log(self.aucroc_keys[i] + '_sc1',
                         np.mean(scores_np_abnormal),
                         prog_bar=True,
                         sync_dist=True)

        if self.visual:
            names = ["h_t", "c_t", "c_tn", "h_tn", 'h_tnn', 'h_tonn']
            count = 0
            # mse_gates = unpack_labels_scores[-3]
            for values in unpack_labels_scores[-8:-2]:
                value_np = torch.stack(values).cpu().data.numpy()
                self.log(names[count],
                         np.mean(value_np[labels_np == 0]),
                         prog_bar=True,
                         sync_dist=True)
                self.log(names[count] + '1',
                         np.mean(value_np[labels_np == 1]),
                         prog_bar=True,
                         sync_dist=True)
                count += 1

        self.validation_step_outputs.clear()


class AeV4V1MsdLstmV4(AeV4V1MsdLstmV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        """add kl divergence with gaussian noise

        Attributes:
            rep_dim: A boolean indicating if we like SPAM or not.
            eggs: An integer count of the eggs we have laid.
        """
        # super().__init__(seed, center, nu, rep_dim, mse_loss_weight, lr,
        #                  weight_decay, lr_milestone, optimizer_name, visual,
        #                  objective)
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")
        self.lstm = MyLSTMV2(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        # self.decoder.eval()
        outputs = self(inputs)
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)
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


class AeV4V1MsdLstmV5(AeV4V1MsdLstmV4):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         kl_loss_weight=kl_loss_weight,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.lstm = MyLSTMV3(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)


class AeV4V1MsdLstmV6(AeV4V1MsdLstmV4):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         kl_loss_weight=kl_loss_weight,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.lstm = MyLSTMV4(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)


class AeV4V1MsdLstmV7(AeV4V1MsdLstmV4):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         kl_loss_weight=kl_loss_weight,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.lstm = MyLSTMV0(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)


class AeV4V1MsdLstmV8(AeV4V1MsdLstmV4):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         kl_loss_weight=kl_loss_weight,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.lstm = MyLSTMV5(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)


class AeV4V1MsdLstmV9(AeV4V1MsdLstmV4):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         kl_loss_weight=kl_loss_weight,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.lstm = MyLSTMV6(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        lstm_x, gates = self.lstm(x)
        # which do harm to performance of the model
        # x = self.bn2d(lstm_x)
        enc_x = self.fc1(lstm_x)
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

        return Ae(dec_out=dec_x, enc_out=enc_x), gates

    def init_center_c(self, net, train_loader, eps=0.1):
        """ init center vector of deep SVDD """
        centers = []
        net = net.cuda()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _ = data
                inputs = inputs.cuda()
                outputs, _ = net(inputs)
                enc_out = outputs.enc_out
                centers.append(enc_out)
        c = torch.mean((torch.cat(centers)), dim=0).cuda()

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.center = c

    def training_step(self, train_batch, batch_idx):
        inputs, _ = train_batch
        # self.decoder.eval()
        outputs, _ = self(inputs)
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
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

        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs, gates = self(inputs)
        mse_gates = []
        for i in range(len(gates)):
            mse_gates.append(
                torch.sum(gates[i], dim=tuple(range(1, gates[i].dim()))))
        h_t, c_t = mse_gates
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        # l1 score
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            dec_out.size(0), -1).sum(dim=1, keepdim=False)
        l1_svdd_scores = enc_out.sub(self.center).abs().contiguous().view(
            enc_out.size(0), -1).sum(dim=1, keepdim=False)
        svdd_mse_scores = (svdd_scores - svdd_scores.min()) / (svdd_scores.max(
        ) - svdd_scores.min()) + (mse_scores - mse_scores.min()) / (
            mse_scores.max() - mse_scores.min())
        # Save triples of (idx, label, score) in a list
        zip_params = [
            labels, svdd_scores, mse_scores, svdd_mse_scores, l1_mse_scores,
            l1_svdd_scores, h_t, c_t
        ]
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
                scores_np_normal = scores_np[labels_np == 0]
                scores_np_abnormal = scores_np[labels_np == 1]
                self.log(self.aucroc_keys[i] + '_sc0',
                         np.mean(scores_np_normal),
                         prog_bar=True,
                         sync_dist=True)
                self.log(self.aucroc_keys[i] + '_sc1',
                         np.mean(scores_np_abnormal),
                         prog_bar=True,
                         sync_dist=True)

        if self.visual:
            names = ["h_t", "c_t"]
            count = 0
            # mse_gates = unpack_labels_scores[-2:]
            for values in unpack_labels_scores[-2:]:
                value_np = torch.stack(values).cpu().data.numpy()
                self.log(names[count],
                         np.mean(value_np[labels_np == 0]),
                         prog_bar=True,
                         sync_dist=True)
                self.log(names[count] + '1',
                         np.mean(value_np[labels_np == 1]),
                         prog_bar=True,
                         sync_dist=True)
                count += 1

        self.validation_step_outputs.clear()


class AeV4V1MsdLstmV10(AeV4V1MsdLstmV1):

    def __init__(self,
                 seed,
                 center=None,
                 nu: float = 0.1,
                 kl_loss_weight=0.1,
                 rep_dim=128,
                 mse_loss_weight=1,
                 lr=0.0001,
                 num_layers=1,
                 weight_decay=0.5e-6,
                 lr_milestones=[250],
                 optimizer_name='amsgrad',
                 visual=False,
                 objective='one-class'):
        super().__init__(seed=seed,
                         center=center,
                         nu=nu,
                         rep_dim=rep_dim,
                         mse_loss_weight=mse_loss_weight,
                         lr=lr,
                         weight_decay=weight_decay,
                         lr_milestones=lr_milestones,
                         num_layers=num_layers,
                         optimizer_name=optimizer_name,
                         visual=visual,
                         objective=objective)
        self.kl_loss_weight = kl_loss_weight
        self.kl_divergence = nn.KLDivLoss(reduction="batchmean")
        self.lstm = MyLSTMV7(self.rep_dim * 4 * 4, self.hidden_layer * 4 * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        lstm_x, _, gates = self.lstm(x)
        # which do harm to performance of the model
        # x = self.bn2d(lstm_x)
        enc_x = self.fc1(lstm_x)
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

        return Ae(dec_out=dec_x, enc_out=enc_x), gates

    def init_center_c(self, net, train_loader, eps=0.1):
        """ init center vector of deep SVDD """
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
                outputs, _ = net(inputs)
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

    def training_step(self, train_batch, batch_idx):
        inputs, _ = train_batch
        # self.decoder.eval()
        outputs, _ = self(inputs)
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        # if self.global_step == 0:
        #     self.logger.experiment.add_graph(self, inputs)
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

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs, gates = self(inputs)
        mse_gates = []
        for i in range(len(gates)):
            # 这里计算是针对input gate和output gate因为logistic sigmoid function
            # 的结果是在0到1之间，但是C就不行，因为tanh是在-1 and 1，这会磨除正负性
            mse_gates.append(
                torch.sum(gates[i], dim=tuple(range(1, gates[i].dim()))))
        i_t, o_t = mse_gates
        enc_out = outputs.enc_out
        dec_out = outputs.dec_out
        dist = torch.sum((enc_out - self.center)**2, dim=1)
        if self.objective == 'soft-boundary':
            svdd_scores = dist - self.R**2
        else:
            svdd_scores = dist
        mse_scores = torch.sum((dec_out - inputs)**2,
                               dim=tuple(range(1, dec_out.dim())))
        # l1 score
        l1_mse_scores = dec_out.sub(inputs).abs().contiguous().view(
            dec_out.size(0), -1).sum(dim=1, keepdim=False)
        l1_svdd_scores = enc_out.sub(self.center).abs().contiguous().view(
            enc_out.size(0), -1).sum(dim=1, keepdim=False)
        svdd_mse_scores = (svdd_scores - svdd_scores.min()) / (svdd_scores.max(
        ) - svdd_scores.min()) + (mse_scores - mse_scores.min()) / (
            mse_scores.max() - mse_scores.min())
        # Save triples of (idx, label, score) in a list
        zip_params = [
            labels, svdd_scores, mse_scores, svdd_mse_scores, l1_mse_scores,
            l1_svdd_scores, i_t, o_t
        ]
        self.validation_step_outputs += list(zip(*zip_params))

    def on_validation_epoch_end(self):
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
                scores_np_normal = scores_np[labels_np == 0]
                scores_np_abnormal = scores_np[labels_np == 1]
                self.log(self.aucroc_keys[i] + '_sc0',
                         np.mean(scores_np_normal),
                         prog_bar=True,
                         sync_dist=True)
                self.log(self.aucroc_keys[i] + '_sc1',
                         np.mean(scores_np_abnormal),
                         prog_bar=True,
                         sync_dist=True)

        if self.visual:
            names = ["i_t", "o_t"]
            count = 0
            for values in unpack_labels_scores[-2:]:
                value_np = torch.stack(values).cpu().data.numpy()
                self.log(names[count],
                         np.mean(value_np[labels_np == 0]),
                         prog_bar=True,
                         sync_dist=True)
                self.log(names[count] + '1',
                         np.mean(value_np[labels_np == 1]),
                         prog_bar=True,
                         sync_dist=True)
                count += 1

        self.validation_step_outputs.clear()
