import torch
import numpy as np
import random
import os
import sys
import yaml
import pandas as pd
import glob
from PIL import Image

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())

from tsne_torch import TorchTSNE as TSNE
import matplotlib.pyplot as plt


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True


def display_tsne(net, dataloader, filename='tsne.png'):
    n_samples = 0
    data_xs = []
    net = net.cuda()
    collect_labels = []
    with torch.no_grad():
        for data in dataloader:
            # get the inputs of the batch
            inputs, labels = data
            inputs = inputs.cuda()
            outputs = net(inputs)
            if isinstance(outputs, dict):
                outputs = outputs["enc_out"]
                outputs = outputs.contiguous().view(outputs.size(0), -1)
            n_samples += outputs.shape[0]
            data_xs.append(outputs)
            collect_labels.append(labels)

    data_x = torch.cat(data_xs, dim=0)
    collect_labels = torch.cat(collect_labels, dim=0).cpu().detach().numpy()
    X_emb = TSNE(n_components=2).fit_transform(data_x.cpu().numpy())
    X_emb = X_emb.cpu().detach().numpy()
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=[collect_labels])
    plt.savefig(filename)


def PCA_svd(X, k, center=True):
    n = X.shape[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) *
         torch.mm(ones, ones.t())) if center else torch.zeros(n *
                                                              n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components.cpu().detach().numpy()


def transfer_weights(dst_net, src_net):
    """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

    dst_net_dict = dst_net.state_dict()
    src_net_dict = src_net.state_dict()

    # Filter out decoder network keys
    src_net_dict = {k: v for k, v in src_net_dict.items() if k in dst_net_dict}
    # Overwrite values in the existing state_dict
    dst_net_dict.update(src_net_dict)
    # Load the new state_dict
    dst_net.load_state_dict(dst_net_dict)


# def load_pre_ae_model(bash_log_name,
#                       batch_size,
#                       radio: float,
#                       n_epochs,
#                       seed,
#                       normal_class,
#                       dataset="cifar10",
#                       objective="ae",
#                       model_name="ae") -> str:
#     """
#     Args:
#     hyperparamters of model

#     Returns:
#     model path
#     """
#     path = "%s/%s/%s/%s/batch_size%d/radio%.2f/n_epochs%d/" % (
#         bash_log_name, dataset, objective, model_name, batch_size, radio,
#         n_epochs)
#     # print(path)
#     keys = ['model_path', 'normal_class', 'seed']
#     hypers_table = pd.DataFrame(columns=keys)

#     mid_path = "lightning_logs/version_0/"
#     time_dirs = sorted(os.listdir(path))
#     for time_dir in time_dirs:
#         lightning_dir = os.path.join(path, time_dir, mid_path)
#         tmp_dict = {"model_path": lightning_dir}
#         with open(os.path.join(lightning_dir, "hparams.yaml"),
#                   'r') as file_handle:
#             hparams = yaml.full_load(file_handle)
#             tmp_dict.update({k: v for k, v in hparams.items() if k in keys})
#             hypers_table = pd.concat(
#                 [hypers_table, pd.DataFrame(tmp_dict, index=[0])],
#                 ignore_index=True)
#     line = hypers_table[(hypers_table['seed'] == seed)
#                         & (hypers_table['normal_class'] == normal_class)]
#     # print(hypers_table)
#     dirs = os.walk(line['model_path'].tolist()[0])
#     for root, _, files in dirs:
#         if root.endswith("checkpoints"):
#             for file in files:
#                 if file.endswith("ckpt"):
#                     return os.path.join(root, file)

#     return None


def load_pre_ae_model(bash_log_name: str,
                      batch_size: int,
                      radio: float,
                      n_epochs: int,
                      seed: int,
                      normal_class: int,
                      dataset="cifar10",
                      objective="ae",
                      model_name="ae",
                      epoch_index=1) -> str:
    """ 
    Args:
    hyperparamters of model

    Returns:
    model path
    """
    path = "%s/%s/%s/%s/batch_size%d/radio%.2f/n_epochs%d/**/lightning_logs/version_0/hparams.yaml" % (
        bash_log_name, dataset, objective, model_name, batch_size, radio,
        n_epochs)
    hparams_files = glob.glob(path, recursive=True)
    # print(hparams_files)
    # print(path)
    keys = ['model_path', 'normal_class', 'seed']
    hypers_table = pd.DataFrame(columns=keys)

    for hparams_file in hparams_files:
        tmp_dict = {"model_path": os.path.dirname(hparams_file)}
        with open(hparams_file, 'r') as file_handle:
            hparams = yaml.full_load(file_handle)
            tmp_dict.update({k: v for k, v in hparams.items() if k in keys})
            hypers_table = pd.concat(
                [hypers_table, pd.DataFrame(tmp_dict, index=[0])],
                ignore_index=True)
    if normal_class is None:
        line = hypers_table[hypers_table['seed'] == seed].iloc[0]
    else:
        line = hypers_table[
            (hypers_table['seed'] == seed)
            & (hypers_table['normal_class'] == normal_class)].iloc[0]

    model_path = sorted(
        glob.glob(os.path.join(line['model_path'], "**/*.ckpt")))
    return model_path[epoch_index - 1]


def min_max(scores, mask):
    psnr_vec = scores[mask]
    psnr_max = torch.max(psnr_vec)
    psnr_min = torch.min(psnr_vec)
    normalized_psnr = (psnr_vec - psnr_min) / (psnr_max - psnr_min)
    scores[mask] = normalized_psnr
    return scores


if __name__ == '__main__':
    # print(
    #     load_pre_ae_model("bash-logv3",
    #                       100,
    #                       0,
    #                       200,
    #                       5,
    #                       7,
    #                       objective="ae",
    #                       model_name="aev1v9"))

    print(
        load_pre_ae_model("bash-logv3",
                          96,
                          0,
                          200,
                          5,
                          7,
                          dataset="mnist",
                          objective="ognet",
                          model_name="ogv1"))
