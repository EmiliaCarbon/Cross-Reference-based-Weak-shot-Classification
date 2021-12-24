import torch.optim.lr_scheduler
from network.simnet import SimNet
from network.gan import Discriminator
from .util import *
from .meters import MatrixMeter, AverageMeter
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
    exp_name = f"{time.strftime('%Y%m%d%H%M', time.localtime())}_dataset={args.dataset_name}_SimNetTrain_usecr=" \
               f"{args.train_with_cr}_cpb={args.class_per_batch}_spc={args.sample_per_class}_bs=" \
               f"{args.batch_size}_balance={args.train_with_balance}"
    logger = Logger(exp_name, True)

    # define network
    torch.cuda.set_device(args.gpu_ids[0])
    simnet = SimNet("train", args.simnet_backbone, True, args.backbone_path).cuda()
    discriminator = Discriminator(simnet.n_feature).cuda()
    logger.log("Networks initiate successfully...")

    # set random seed
    set_seeds(args.seed)

    data_supplier = get_supplier(args)
    # four dataset, the former two is for training and the latter two is for testing
    train_clean_data = data_supplier.clean_dataset_base("train")
    train_noisy_data = data_supplier.noisy_dataset_novel()
    base_test_data = data_supplier.clean_dataset_base("test")
    novel_test_data = data_supplier.clean_dataset_novel("test")

    # data loader
    base_test_dataloader = valid_balance_dataloader(base_test_data)
    novel_test_dataloader = valid_balance_dataloader(novel_test_data)
    if args.train_with_balance:
        train_clean_dataloader = train_balance_dataloader(train_clean_data, args.class_per_batch)
        train_noisy_dataloader = train_balance_dataloader(train_noisy_data, 1)
    else:
        train_clean_dataloader = train_unbalance_dataloader(train_clean_data, args.class_per_batch)
        train_noisy_dataloader = train_unbalance_dataloader(train_noisy_data, 1)
    logger.log("Dataset initiate successfully...")

    # define optimizer
    simnet_optimizer = optim.SGD([{'params': simnet.parameters(), 'initial_lr': args.lr}],
                                 lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    simnet_lr_schedule = optim.lr_scheduler.LambdaLR(simnet_optimizer, lr_lambda, last_epoch=29)
    gan_optimizer = optim.SGD([{'params': discriminator.parameters(), 'initial_lr': args.lr}],
                              lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    gan_lr_schedule = optim.lr_scheduler.LambdaLR(gan_optimizer, lr_lambda, last_epoch=29)

    # define criterion
    cls_criterion = torch.nn.CrossEntropyLoss()
    gan_criterion = torch.nn.CrossEntropyLoss()
    logger.log("Start training...")

    test_meters = []
    iter_count = 0
    for epoch in range(args.max_epoch):
        logger.log(f"The training processes to {epoch} epoch...")
        logger.log(f"Learning rate in this epoch is {gan_lr_schedule.get_last_lr()}")
        simnet.train()

        train_meter = MatrixMeter(['Dissimilarity', 'Similarity'], default_metric='f1score')
        gan_meter = MatrixMeter(["NoisySet", "CleanSet"], default_metric="acc")
        cls_meter = AverageMeter(f"Average cls loss of epoch {epoch}")

        logger.log(f"Total iteration in this epoch is {len(train_clean_dataloader)}")

        for i, (clean_data, noisy_data) in enumerate(zip(train_clean_dataloader, train_noisy_dataloader)):
            start_time = time.time()

            clean_images, clean_labels, _ = clean_data
            noisy_images, noisy_labels, _ = noisy_data
            clean_images, clean_labels = clean_images[0].cuda(), clean_labels[0].cuda()
            noisy_images, noisy_labels = noisy_images[0].cuda(), noisy_labels[0].cuda()
            n = clean_images.shape[0]
            if clean_images.shape != noisy_images.shape:
                logger.log(f"Iteration {i} skipped because the different batch size of A and B")
                continue
            # define classification target
            cls_target = generate_target(clean_labels).cuda()  # (N*N,)

            # simnet forward
            feature_map = simnet.backbone(torch.cat([clean_images, noisy_images], dim=0))
            feature_clean, feature_noisy = feature_map.chunk(2)
            feature_clean, feature_noisy = simnet.enum(feature_clean), simnet.enum(feature_noisy)
            if args.train_with_cr:
                feature_map = simnet.cross_ref(torch.cat([feature_clean, feature_noisy], dim=0))
            else:
                feature_map = torch.cat([feature_clean, feature_noisy], dim=0)
            con_cls_feature, con_cls_pred = simnet.classifier(simnet.fusion(feature_map))
            # cls_pred has shape (N*N, 2)
            cls_pred, _ = con_cls_pred.chunk(2)

            # SimNet classification loss
            cls_loss = cls_criterion(cls_pred, cls_target)
            # update meter
            train_meter.update(cls_pred, cls_target)
            cls_meter.update(cls_loss.cpu().detach().item())

            # GAN target, 1 is from the clean set, 0 is from the noisy set
            gan_target = torch.cat([torch.ones(n * n), torch.zeros(n * n)]).long().cuda()
            gan_pred = discriminator(con_cls_feature)
            gan_loss = gan_criterion(gan_pred, gan_target)
            # update meter
            gan_meter.update(gan_pred, gan_target)

            # train SimNet
            simnet_optimizer.zero_grad()
            simnet_loss = cls_loss - args.beta * gan_loss
            simnet_loss.backward()
            simnet_optimizer.step()

            # train GAN
            gan_optimizer.zero_grad()
            gan_pred = discriminator(con_cls_feature.clone().detach())
            gan_loss = gan_criterion(gan_pred, gan_target)
            gan_loss.backward()
            gan_optimizer.step()

            iter_count += 1
            end_time = time.time()
            if iter_count % args.log_per_iter == 0:
                iter_loss = cls_loss.cpu().detach().item()
                logger.log(f"{epoch}-{i}: {iter_loss}  time cost: {end_time - start_time}s")
                vis.line(X=np.array([int(iter_count / args.log_per_iter)]), Y=np.array([iter_loss]),
                         win="iteration", update="append",
                         opts=dict(title="iteration"))

        simnet_lr_schedule.step()
        gan_lr_schedule.step()

        val_meter = valid(simnet, base_test_dataloader, logger.log(f"Report of validation result at epoch {epoch}:"))
        test_meter = valid(simnet, novel_test_dataloader, logger.log(f"Report of test result at epoch {epoch}:"))
        test_meters.append(test_meter.get_main())

        save_model_if_best(test_meters, simnet,
                           os.path.join(args.best_save_path, exp_name, f"simnet_best_value.pth"), logger)
        save_model_if_best(test_meters, discriminator,
                           os.path.join(args.best_save_path, exp_name, f"discrim_best_value.pth"), logger)

        if (epoch + 1) % args.save_every_epoch == 0:
            if not os.path.isdir(os.path.join(args.auto_save_path, exp_name)):
                os.mkdir(os.path.join(args.auto_save_path, exp_name), 0o777)
            torch.save(simnet.state_dict(), os.path.join(args.auto_save_path, exp_name, f"simnet_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(),
                       os.path.join(args.auto_save_path, exp_name, f"discrim_epoch_{epoch}.pth"))

        log_and_visualize(logger, vis, epoch,
                          [train_meter, cls_meter, val_meter, test_meter],
                          ["train_f1_score", "train_acc", "valid_f1_score", "test_f1_score"])


def valid(model, dataloader, logger):
    meter = MatrixMeter(['Dissimilarity', 'Similarity'], default_metric='f1score')
    model.eval()

    with torch.no_grad():
        for _, (images, classes, _) in tqdm(enumerate(dataloader)):
            cls_pred, _ = model(images.cuda())
            cls_target = generate_target(classes).cuda()
            meter.update(cls_pred, cls_target)
    logger.log(meter.report())
    return meter
