import os.path

import torch.optim.lr_scheduler
from network.classifier import Classifier
from network.simnet import SimNet
from .util import *
from .meters import MatrixMeter
from torch import optim
from tqdm import tqdm
import threading
import time


def train():
    # sub-thread to active visdom server
    vis_thread = threading.Thread(target=visdom_server)
    vis_thread.start()
    time.sleep(2)
    vis = Visdom(server="http://localhost", env=args.env, port=args.visdom_port)

    # logger
    exp_name = f"{time.strftime('%Y%m%d%H%M', time.localtime())}_dataset={args.dataset_name}_ClassifierTrain_bs=" \
               f"{args.classifier_batch_size}_usecrweight={args.train_with_cr_weight}_labelsmo=" \
               f"{args.label_smooth}_lamb={args.lamb}_lr={args.cls_lr}"
    logger = Logger(exp_name, True)

    test_transforms = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # define dataset
    data_supplier = get_supplier(args)
    train_data = data_supplier.noisy_dataset_novel()
    train_data.transform = test_transforms
    test_data = data_supplier.clean_dataset_novel("test")
    # data loader
    train_loader = DataLoader(train_data, batch_size=args.classifier_batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.classifier_batch_size,
                             shuffle=False, num_workers=args.num_workers)
    logger.log("Dataset initiate successfully...")

    # define network
    torch.cuda.set_device(args.gpu_ids[0])
    model = Classifier(args.simnet_backbone, len(train_data.classes), True, args.backbone_path,
                       map_location=lambda storage, loc: storage.cuda()).cuda()
    simnet = SimNet("infer", args.simnet_backbone, False, None).cuda()
    simnet.load_state_dict(torch.load(args.pretrained_simnet_path, map_location=lambda storage, loc: storage.cuda()))
    simnet.parallel(args.gpu_ids)
    logger.log("Networks initiate successfully...")

    # set random seed
    set_seeds(args.seed)

    # define optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=args.cls_lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # load the weights file
    weights_dict = load_weights_dict(args.weight_load_path, train_data.classes, train_data)
    logger.log("Weights file load successfully...")

    # define criterion
    criterion = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smooth)
    logger.log("Start training...")

    test_meters = []
    iter_count = 0

    for epoch in range(args.max_epoch):
        logger.log(f"The training processes to {epoch} epoch...")
        logger.log(f"Learning rate in this epoch is {scheduler.get_last_lr()}")
        model.train()

        train_meter = MatrixMeter(train_data.classes, default_metric='acc')

        logger.log(f"Total iteration in this epoch is {len(train_loader)}")

        for i, (images, classes, paths) in enumerate(train_loader):
            start_time = time.time()
            images = images.cuda()
            # get the cross-reference weight and similarity weight
            if args.train_with_cr_weight:
                cr_weights, sim_weights = get_weights(weights_dict, classes.numpy().tolist(), list(paths))
                feat, pred = model(images, cr_weights.cuda())
            else:
                sim_weights = get_weights(weights_dict, classes.numpy().tolist(), list(paths))
                feat, pred = model(images, None)
            classes = classes.cuda()
            loss_cls = torch.mean(criterion(pred, classes) * sim_weights.cuda())        # classification loss

            # calculate the reg loss
            n_sample, n_channel = feat.shape
            with torch.no_grad():
                simnet.eval()
                sim, _ = simnet(images)
                sim_matrix = torch.softmax(sim, dim=1)[:, 1].reshape(n_sample, n_sample).detach()

            feat_0 = feat.repeat(1, n_sample).reshape(n_sample, n_sample, n_channel)
            feat_1 = feat.repeat(n_sample, 1).reshape(n_sample, n_sample, n_channel)
            loss_reg = torch.sum(torch.sum(torch.square(feat_0 - feat_1), dim=2) * sim_matrix)

            loss = loss_cls + 0.1 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(pred, classes)

            iter_count += 1
            end_time = time.time()
            if iter_count % args.log_per_iter == 0:
                iter_loss = loss.cpu().detach().item()
                logger.log(f"{epoch}-{i}: {iter_loss}  time cost: {end_time - start_time}s")
                vis.line(X=np.array([int(iter_count / args.log_per_iter)]), Y=np.array([iter_loss]),
                         win="iteration", update="append",
                         opts=dict(title="iteration"))

        scheduler.step()

        test_meter = valid(model, test_loader, logger.log(f"Report of test result at epoch {epoch}:"))
        test_meters.append(test_meter.get_main())

        save_model_if_best(test_meters, model,
                           os.path.join(args.best_save_path, exp_name, f"{exp_name}_classifier_optim_best.pth"), logger)

        if (epoch + 1) % args.save_every_epoch == 0:
            if not os.path.isdir(os.path.join(args.auto_save_path, exp_name)):
                os.mkdir(os.path.join(args.auto_save_path, exp_name), 0o777)
            torch.save(model.state_dict(),
                       os.path.join(args.auto_save_path, exp_name, f"{exp_name}_classifier_{epoch}.pth"))

        log_and_visualize(logger, vis, epoch,
                          [train_meter, test_meter], ["train_acc", "test_acc"])


def get_weights(weights_dict, classes: List[int], paths: List[str]):
    """
    :return: cr_weights: Tensor (N, C, H, W) if exists
             sim_weights: Tensor (N,)
    """
    cr_weights, sim_weights = [], []
    for i in range(len(classes)):
        cls_weight = weights_dict[classes[i]]
        index = cls_weight["names"].index(os.path.split(paths[i])[-1])
        if "cr_weights" in cls_weight.keys():
            cr_weights.append(cls_weight["cr_weights"][index])
        sim_weights.append(cls_weight["sim_weights"][index])
    if args.train_with_cr_weight:
        return torch.stack(cr_weights), torch.stack(sim_weights)
    return torch.stack(sim_weights)


def load_weights_dict(root_path, classes: List[str], dataset: BaseDataset):
    weights_dict = {}
    for cls in tqdm(classes):
        weights = torch.load(os.path.join(root_path, f"{cls}_weights.pth"),
                             map_location="cpu")
        paths = weights["names"]
        for i in range(len(paths)):
            paths[i] = os.path.split(paths[i])[-1]
        weights["names"] = paths
        weights["sim_weights"] = torch.from_numpy(sim_matrix_to_weight(weights["sim_matrix"].numpy(), args.lamb))
        del weights["sim_matrix"]
        weights_dict[dataset.cls2int[cls]] = weights
    return weights_dict


def valid(model, dataloader, logger):
    meter = MatrixMeter(dataloader.dataset.classes, default_metric='acc')
    model.eval()

    with torch.no_grad():
        for _, (images, classes, _) in tqdm(enumerate(dataloader)):
            _, pred = model(images.cuda())
            meter.update(pred, classes)
    logger.log(meter.report())
    return meter
