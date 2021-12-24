import os.path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from data.base_dataset import BaseDataset
from .simnet import SimNet, SingleEnumerator
from torchvision.transforms import transforms
from config import args


def get_and_save_weight(dataset: BaseDataset, model: SimNet, exp_name):
    image_dict = {}
    for path, cls in dataset.image_list:
        if cls in image_dict.keys():
            image_dict[cls].append(path)
        else:
            image_dict[cls] = [path]

    test_transforms = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    step_size = 50
    enum = SingleEnumerator()
    root_path = os.path.join(args.weight_save_path, exp_name)
    model.eval()
    with torch.no_grad():
        for cls, path_list in tqdm(image_dict.items()):
            feature_maps = []
            batch_num = np.ceil(len(path_list) / step_size)
            for i in range(batch_num):
                paths = path_list[i * step_size: (i + 1) * step_size]
                images = [test_transforms(Image.open(path).convert('RGB')) for path in paths]
                feature_maps.append(model.backbone(torch.stack(images).cuda()))
            feature_maps = torch.cat(feature_maps, dim=0)  # feature_maps has shape (N, C, H, W)
            sample_size = feature_maps.shape[0]

            cr_weights = []
            sim_matrix = torch.zeros(sample_size, sample_size)
            for i in range(sample_size):
                enum_feature = enum(feature_maps[i], feature_maps)  # enum_feature has shape (N, 2, C, H, W)
                cr_feature = model.cross_ref(enum_feature)
                cr_weight = torch.mean(cr_feature[:, 0, ...])
                cr_weights.append(cr_weight.cpu())

                _, sim = model.classifier(model.fusion(cr_feature))
                sim_matrix[i] = torch.softmax(sim, dim=1)[:, 1].cpu()
            cr_weights = torch.stack(cr_weight, dim=0)
            sim_weights = torch.from_numpy(sim_matrix_to_weight(sim_matrix.numpy(), args.lamb))
            weight = {
                "cr_weights": cr_weights,       # (N, C, H, W)
                "sim_weights": sim_weights       # (N,)
            }
            torch.save(weight, os.path.join(root_path, f"{dataset.int2cls[cls]}_weights.pth"))


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
