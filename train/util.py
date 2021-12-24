import random
from config import args
import subprocess
from subprocess import DEVNULL
from typing import List, Union
from visdom import Visdom
from .meters import *
import glob
import os
from torch.utils.data import DataLoader
from data.base_dataset import *


def valid_balance_dataloader(dataset):
    return DataLoader(dataset,
                      batch_sampler=BalancedBatchSampler(dataset,
                                                         args.class_per_batch, args.sample_per_class, args.batch_num),
                      batch_size=1, num_workers=args.num_workers)


def train_unbalance_dataloader(dataset, class_per_batch):
    rand_sample = RandBatchDataset(dataset, class_per_batch, args.batch_size)
    return DataLoader(rand_sample, batch_size=1, num_workers=args.num_workers)


def train_balance_dataloader(dataset, class_per_batch):
    balance_sample = BalancedBatchDataset(dataset, class_per_batch, args.batch_size)
    return DataLoader(balance_sample, batch_size=1, num_workers=args.num_workers)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(epoch):
    return args.lr_decay ** (epoch // args.lr_interval)


def generate_target(y: torch.Tensor):
    """
    generate the ground-truth target of y
    :param y: Tensor with shape (N,)
    :return: Tensor with shape (N*N,)
    """
    assert y.ndim == 1
    n = len(y)
    y = torch.unsqueeze(y, dim=1)
    y = torch.cat([y.repeat(1, n).view(-1, 1), y.repeat(n, 1)], dim=1)
    return (y[:, 0] == y[:, 1]).long()


def visdom_server():
    """
    start visdom service
    """
    subprocess.run(["/home/suqi/anaconda3/envs/torch/bin/python",
                    "-m", "visdom.server", "-port", f"{args.visdom_port}"],
                   stdout=DEVNULL, stderr=DEVNULL)


def log_and_visualize(logger, vis: Visdom, epoch, meters: List[Union[AverageMeter, MatrixMeter]], names: List[str]):
    assert len(meters) == len(names)
    logger.log(f"<-------------------epoch {epoch} finished------------------->")
    for i in range(len(names)):
        if isinstance(meters[i],  MatrixMeter):
            value = meters[i].get_main()
        else:
            value = meters[i].avg
        logger.log(f"{names[i]}: {value:3.1%}")
        vis.line(X=np.array([epoch]), Y=np.array([value]),
                 win=names[i], update="append",
                 opts=dict(title=names[i]))


class Logger:
    def __init__(self, exp_name, print_std=True):
        exist_files = glob.glob(os.path.join(args.log_root_path, f"{exp_name}_*.log"))
        try:
            index = np.max([int(os.path.split(file)[-1].split(f"{exp_name}_")[1].split(".")[0])
                            for file in exist_files]) + 1
        except ValueError:
            index = 0
        self.log_file = os.path.join(args.log_root_path, f"{exp_name}_{index}.log")
        self.print_std = print_std
        with open(self.log_file, "x"):
            pass

    def log(self, info):
        with open(self.log_file, "a") as f:
            f.write(str(info) + "\n")
        if self.print_std:
            print(info)
        return self


def save_model_if_best(test_meters: list, model, path, logger):
    if test_meters[-1] < 0.4:
        return
    epoch = len(test_meters) - 1
    best_epoch = np.argmax(test_meters)

    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), mode=0o777)

    if epoch == best_epoch:
        torch.save(model.state_dict(), path.replace('value', f'{test_meters[-1]:.4f}'))
        logger.log(f"best model saved at: \"{path}\"")

    else:
        logger.log(f"best: {best_epoch}, current: {epoch}")


def sim_matrix_to_weight(sim_matrix: np.ndarray, lamb) -> np.ndarray:
    assert sim_matrix.ndim == 2
    batch_size = sim_matrix.shape[0]
    density = np.zeros(shape=(batch_size,))
    prefix = 1 / (batch_size * np.sqrt(2 * np.pi * lamb ** 2))
    for i in range(batch_size):
        density[i] = prefix * np.sum([np.exp(- sample_dist(sim_matrix, i, j) ** 2 / (2 * lamb ** 2))
                                     for j in range(batch_size)])
    return density / (np.sum(density) / batch_size)


def sample_dist(sim_matrix, i, j):
    return 1 - (sim_matrix[i, j] + sim_matrix[j, i]) / 2
