from PIL import Image
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from torchvision.transforms import transforms


class SupplierBase:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.root_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.base_classes, self.novel_classes = [], []
        self.train_transforms = transforms.Compose(
            [transforms.RandomRotation(30), transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
             transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.test_transforms = transforms.Compose(
            [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def clean_dataset_base(self, mode):
        raise NotImplementedError

    def clean_dataset_novel(self, mode):
        raise NotImplementedError

    def noisy_dataset_novel(self):
        raise NotImplementedError


class BaseDataset:
    def __init__(self, root_path, classes, transform, image_list=None):
        self.root_path = root_path
        self.classes = classes

        self.cls2int = {classes[i]: i for i in range(len(classes))}
        self.int2cls = {i: classes[i] for i in range(len(classes))}

        self.transform = transform

        if image_list is None:
            self.image_list = []
        else:
            self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path, cls = self.image_list[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, cls, img_path

    def load(self, *args):
        raise NotImplementedError

    def split(self, rate):
        """
        split the dataset into train set and validation set by rate.
        the train set and validation set have the same classes, and the samples of each class are separated
        at a proportion of rate.
        """
        train_list, val_list = [], []
        image_dict = {}
        for key in self.int2cls.keys():
            image_dict[key] = []
        for path, cls in self.image_list:
            image_dict[cls].append(path)

        for cls, paths in image_dict.items():
            val_length = int(len(paths) * rate)
            train_list += [(path, cls) for path in paths[val_length:]]
            val_list += [(path, cls) for path in paths[:val_length]]

        train_set = BaseDataset(self.root_path, self.classes, self.transform, train_list)
        val_set = BaseDataset(self.root_path, self.classes, self.transform, val_list)
        return train_set, val_set


# randomly sample from the dateset, every batch has class_per_batch classes
class RandBatchDataset(Dataset):
    def __init__(self, dataset: BaseDataset, class_per_batch, batch_size):
        self.dataset = dataset
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size
        self.batches = []
        self.sample()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        images, classes, paths = [], [], []
        for i in self.batches[index]:
            img, cls, path = self.dataset[i]
            images.append(img)
            classes.append(cls)
            paths.append(path)

        return torch.stack(images, dim=0), torch.tensor(classes), paths

    def sample(self):
        data_dict = {}
        for i, (path, cls) in enumerate(self.dataset.image_list):
            if cls in data_dict.keys():
                data_dict[cls].append(i)
            else:
                data_dict[cls] = [i]

        length = len(self.dataset) // self.batch_size

        for i in range(length):
            batch, cls_left = [], []
            for cls, indexes in data_dict.items():
                if len(indexes) > 0:
                    cls_left.append(cls)

            if len(cls_left) == 0:
                break

            chosen_classes = np.random.permutation(cls_left).tolist()[:self.class_per_batch]

            for _ in range(self.batch_size):
                if not chosen_classes:
                    break

                chosen_cls = np.random.choice(chosen_classes, 1)[0]
                cls_index_num = len(data_dict[chosen_cls])

                batch.append(data_dict[chosen_cls].pop(np.random.randint(cls_index_num)))
                if cls_index_num == 1:
                    chosen_classes.remove(chosen_cls)

            self.batches.append(batch)


# randomly sample from the dateset with class balance
class BalancedBatchDataset(Dataset):
    def __init__(self, dataset: BaseDataset, class_per_batch, batch_size):
        self.dataset = dataset
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size
        self.sample_per_class = int(batch_size / class_per_batch)
        self.batches = []
        self.sample()

    def sample(self):
        data_dict = {}
        for i, (path, cls) in enumerate(self.dataset.image_list):
            if cls in data_dict.keys():
                data_dict[cls].append(i)
            else:
                data_dict[cls] = [i]

        length = len(self.dataset) // self.batch_size
        for i in range(length):
            batch, cls_left = [], []
            for cls, indexes in data_dict.items():
                if len(indexes) >= self.sample_per_class:
                    cls_left.append(cls)
            if len(cls_left) < self.class_per_batch:
                break
            chosen_classes = np.random.permutation(cls_left).tolist()[:self.class_per_batch]
            for cls in chosen_classes:
                chosen_samples = np.random.choice(data_dict[cls], self.sample_per_class, False).tolist()
                batch.extend(chosen_samples)
                for sample in chosen_samples:
                    data_dict[cls].remove(sample)
                if len(data_dict[cls]) < self.sample_per_class:
                    data_dict.pop(cls)
            self.batches.append(batch)

    def __getitem__(self, index):
        images, classes, paths = [], [], []
        for i in self.batches[index]:
            img, cls, path = self.dataset[i]
            images.append(img)
            classes.append(cls)
            paths.append(path)

        return torch.stack(images, dim=0), torch.tensor(classes), paths

    def __len__(self):
        return len(self.batches)


# randomly sample from the dateset, every batch has indefinite classes, every class only one sample in one batch
class SingleBatchDataset(Dataset):
    def __init__(self, dataset, batch_size, max_length=5000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.batches = []
        self.sample()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        images, classes, paths = [], [], []
        for i in self.batches[index]:
            img, cls, path = self.dataset[i]
            images.append(img)
            classes.append(cls)
            paths.append(path)

        return torch.stack(images, dim=0), torch.tensor(classes), paths

    def sample(self):
        batches = []
        data_dict = {}
        for i, (path, cls) in enumerate(self.dataset.image_list):
            if cls in data_dict.keys():
                data_dict[cls].append(i)
            else:
                data_dict[cls] = [i]
        class_num = len(data_dict.keys())
        for i in range(self.max_length):
            batch = []
            chosen_classes = np.random.choice(list(data_dict.keys()), min(self.batch_size, class_num), replace=False)
            for cls in chosen_classes:
                batch.append(np.random.choice(data_dict[cls], 1)[0])

            batches.append(batch)
        self.batches = batches


class FuseDataset(Dataset):
    def __init__(self, dataset_0, dataset_1):
        self.dataset_0 = dataset_0
        self.dataset_1 = dataset_1

    def __len__(self):
        return len(self.dataset_0) + len(self.dataset_1)

    def __getitem__(self, index):
        if index < len(self.dataset_0):
            return self.dataset_0[index]
        else:
            return self.dataset_1[index - len(self.dataset_0)]


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset: BaseDataset, class_per_batch, sample_per_class, batch_num=10):
        super(BalancedBatchSampler, self).__init__(dataset)

        image_dict = {}
        for cls in dataset.int2cls.keys():
            image_dict[cls] = []
        for idx, (p, c) in enumerate(dataset.image_list):
            image_dict[c].append(idx)

        self.batches = []

        for _ in range(batch_num):
            batch = []
            chosen_classes = np.random.choice(len(image_dict), class_per_batch, replace=False)
            for cls in chosen_classes:
                image_idx = image_dict[cls]
                chosen_index = np.random.choice(image_idx, sample_per_class,
                                                replace=len(image_idx) < sample_per_class).tolist()
                batch += chosen_index
            self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


from .cub_dataset import CUBSupplier
from .car_dataset import CarSupplier
from .air_dataset import AirSupplier


def get_supplier(args) -> SupplierBase:
    if args.dataset_name == "CUB":
        return CUBSupplier(args)
    elif args.dataset_name == "Car":
        return CarSupplier(args)
    elif args.dataset_name == "Air":
        return AirSupplier(args)
    else:
        raise ValueError(f"dataset name \"{args.dataset_name}\" is not defined")
