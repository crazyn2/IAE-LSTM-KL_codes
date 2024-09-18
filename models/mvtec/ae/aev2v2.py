from torch import nn
from models.base.ae import Ae
from models.mvtec.base import MvTecAeV2


class EncoderV2(nn.Module):

    def __init__(self, chnum_in, feature_num, feature_num_2,
                 feature_num_x2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(chnum_in,
                      feature_num_2,
                      bias=False,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(feature_num_2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2,
                      feature_num,
                      bias=False,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(feature_num, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num,
                      feature_num_x2,
                      bias=False,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(feature_num_x2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2,
                      feature_num_x2,
                      bias=False,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.BatchNorm2d(feature_num_x2, affine=False),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, input_x):
        return self.encoder(input_x)


class DecoderV2(nn.Module):

    def __init__(self, chnum_in, feature_num, feature_num_2,
                 feature_num_x2) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_num_x2,
                               feature_num_x2,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(feature_num_x2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_x2,
                               feature_num,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(feature_num, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num,
                               feature_num_2,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(feature_num_2, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_2,
                               chnum_in,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1,
                               output_padding=1))

    def forward(self, input_x):
        return self.decoder(input_x)


class AeV2V2V1(MvTecAeV2):

    def __init__(
        self,
        seed,
        chnum_in=3,
        lr=0.0001,
        weight_decay=0.5e-6,
        lr_milestones=[250],
        optimizer_name='amsgrad',
        visual=False,
    ):
        super().__init__(
            seed=seed,
            lr=lr,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            optimizer_name=optimizer_name,
            visual=visual,
        )
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = EncoderV2(chnum_in, feature_num, feature_num_2,
                                 feature_num_x2)

        self.decoder = DecoderV2(chnum_in, feature_num, feature_num_2,
                                 feature_num_x2)

        self.mse = nn.MSELoss(reduction='mean')
        self.validation_step_outputs = []

    def forward(self, x):
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)
        return Ae(dec_out=dec_x, enc_out=enc_x)
