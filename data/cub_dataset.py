from .base_dataset import BaseDataset, SupplierBase
import os


def read_split_from_file(file_path):
    classes = []
    for line in open(file_path).readlines():
        classes.append(line.strip())
    return classes


class CUBSupplier(SupplierBase):
    def __init__(self, args):
        super(CUBSupplier, self).__init__(args)
        self.base_classes = read_split_from_file("data/CUB/train_classes.txt")
        self.novel_classes = read_split_from_file("data/CUB/test_classes.txt")

    def clean_dataset_base(self, mode):
        assert mode in ["train", "test"]
        transform = self.train_transforms if mode == "train" else self.test_transforms
        return CUBDataset(mode, self.root_path, self.base_classes, transform)

    def clean_dataset_novel(self, mode):
        assert mode in ["train", "test"]
        transform = self.train_transforms if mode == "train" else self.test_transforms
        return CUBDataset(mode, self.root_path, self.novel_classes, transform)

    def noisy_dataset_novel(self):
        return WebSet(self.root_path, self.novel_classes, self.train_transforms)


class CUBDataset(BaseDataset):
    def __init__(self, mode, root_path, classes, transform):
        super(CUBDataset, self).__init__(root_path, classes, transform, None)
        assert mode in ["train", "test"]
        self.load(mode)

    def load(self, mode="train"):
        accept_ids = []
        accept_mode = "1" if mode == "train" else "0"
        for line in open(os.path.join(self.root_path, 'CUB_200_2011', 'train_test_split.txt')).readlines():
            img_id, is_train = line.strip().split(' ')
            if is_train == accept_mode:
                accept_ids.append(img_id)

        for line in open(os.path.join(self.root_path, 'CUB_200_2011', 'images.txt')).readlines():
            img_id, img_path = line.strip().split(' ')
            cls_name, img_name = img_path.split('/')
            if img_id in accept_ids:
                if cls_name in self.classes:
                    self.image_list.append((os.path.join(self.root_path, 'CUB_200_2011', 'images', img_path),
                                            self.cls2int[cls_name]))


class WebSet(BaseDataset):
    def __init__(self, root_path, classes, transform=None):
        super(WebSet, self).__init__(root_path, classes, transform)
        self.load()

    def load(self):
        for cls in self.classes:
            dir_path = os.path.join(self.root_path, 'CUB_web', cls)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"directory \"{dir_path}\" is not found")

            images = sorted(os.listdir(dir_path))
            cls_list = [(os.path.join(dir_path, img), self.cls2int[cls]) for img in images]

            self.image_list += cls_list
