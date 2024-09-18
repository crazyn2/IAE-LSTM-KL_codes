import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.msd import BaseMsdV1
from utils import get_radius
from models.base.ae import Ae


class AeV1V3MsdV1(BaseMsdV1):

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
        """vanilla SVDD autoencoder
        """
        super().__init__(seed, center, nu, rep_dim, mse_loss_weight, lr,
                         weight_decay, lr_milestones, optimizer_name, visual,
                         objective)
        # 考虑下修改神经网络初始化方式
        # https://zhuanlan.zhihu.com/p/405752148
        # LSTM输入输出大小是否会影响结果
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
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
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
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

        # return Dict({"dec_out": dec_x, 'enc_out': enc_x})
        return Ae(dec_out=dec_x, enc_out=enc_x)
