import os.path

from tqdm import tqdm
from network.simnet import SimNet, SingleEnumerator
from train.util import *
import time


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

    step_size_0 = 500
    step_size_1 = 2
    enum = SingleEnumerator()
    root_path = os.path.join(args.weight_save_path, exp_name)
    if not os.path.isdir(root_path):
        os.mkdir(root_path, 0o777)
    model.eval()
# split=[0, 10, 20, 30, 40, 50]
# gpus=[2, 3, 5, 6, 7]
    with torch.no_grad():
        for cls, path_list in tqdm(list(image_dict.items())[40: 50]):
            feature_maps = []
            batch_num = int(np.ceil(len(path_list) / step_size_0))
            for i in tqdm(range(batch_num)):
                paths = path_list[i * step_size_0: (i + 1) * step_size_0]
                images = [test_transforms(Image.open(path).convert('RGB')) for path in paths]
                feature_maps.append(model.backbone(torch.stack(images).cuda()))
            feature_maps = torch.cat(feature_maps, dim=0)  # feature_maps has shape (N, C, H, W)
            sample_size = feature_maps.shape[0]

            cr_weights = []
            sim_matrix = []
            batch_num = int(np.ceil(len(path_list) / step_size_1))
            for i in tqdm(range(batch_num)):
                enum_feature = enum(feature_maps[i * step_size_1: (i + 1) * step_size_1],
                                    feature_maps)  # enum_feature has shape (step_size_1 * N, 2, C, H, W)
                _, _, channel, height, weight = enum_feature.shape
                cr_feature = model.cross_ref(enum_feature)
                temp = torch.reshape(cr_feature, [-1, sample_size, 2, channel, height, weight])
                cr_weight = torch.mean(temp[:, :, 0, ...], dim=1)     # (step_size_1, C, H, W)
                cr_weights.append(cr_weight.cpu())

                _, sim = model.classifier(model.fusion(cr_feature))
                sim = torch.reshape(sim, [-1, sample_size, 2])
                sim_matrix.append(torch.softmax(sim, dim=2)[..., 1].cpu())
            cr_weights = torch.cat(cr_weights, dim=0)
            sim_matrix = torch.cat(sim_matrix, dim=0)
            assert cr_weights.shape[0] == sample_size
            assert sim_matrix.shape == (sample_size, sample_size)
            weight = {
                "cr_weights": cr_weights,       # (N, C, H, W)
                "sim_matrix": sim_matrix,       # (N, N)
                "names": path_list              # (N,)
            }
            torch.save(weight, os.path.join(root_path, f"{dataset.int2cls[cls]}_weights.pth"))


def train():
    torch.cuda.set_device(args.gpu_ids[0])
    data_supplier = get_supplier(args)
    train_data = data_supplier.noisy_dataset_novel()
    model = SimNet("train", "resnet-50", False, None).cuda()
    model.load_state_dict(torch.load(args.pretrained_simnet_path, map_location=lambda storage, loc: storage.cuda(0)))
    # model.parallel(args.gpu_ids)
    exp_name = f"{time.strftime('%Y%m%d%H%M', time.localtime())}_dataset={args.dataset_name}_weights"
    get_and_save_weight(train_data, model, exp_name)
