import argparse

from Config import cfg
from Config import update_config

from utils import create_logger
from utils import save_checkpoint
from model import Sparse_alignment_network
from Dataloader import eCelebV_Dataset, WFLWV_Dataset
from backbone import Alignment_Loss
from utils import get_optimizer
from tools import train
from tools import validate

from tensorboardX import SummaryWriter
import wandb

import torch
import pprint
import os

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Checkpoint')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--target', help='targeted branch (alignment, emotion or pose)',
                        type=str, default='alignment')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def main_function():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # WandB 초기화
    wandb.init(
        project="SLPT_application",
        name=cfg.MODEL.NAME,
        config={
            "Input": cfg.MODEL.IMG_SIZE,
            "Backbone": cfg.MODEL.PRETRAINED,
            "Data": cfg.DATASET.DATASET,
            "Representation": cfg.DATASET.REPR,
            "Batch": cfg.TRAIN.BATCH_SIZE_PER_GPU,
            "Epoch": cfg.TRAIN.NUM_EPOCH,
            "LR": cfg.TRAIN.LR,
            "Optimizer": cfg.TRAIN.OPTIMIZER
        }
    )

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if "eCelebV" in cfg.DATASET.DATASET:
        model = Sparse_alignment_network(cfg.eCelebV.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.eCelebV.INITIAL_PATH, cfg)
    elif cfg.DATASET.DATASET == "WFLWV":
        model = Sparse_alignment_network(cfg.WFLWV.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.WFLWV.INITIAL_PATH, cfg)
    else:
        raise ValueError('Wrong Dataset')
    torch.cuda.set_device(torch.device(f'cuda:{cfg.GPUS[0]}'))
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    loss_function_2 = Alignment_Loss(cfg).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if "eCelebV" in cfg.DATASET.DATASET:
        train_dataset = eCelebV_Dataset(
            cfg, cfg.eCelebV.ROOT, 'train',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_dataset = eCelebV_Dataset(
            cfg, cfg.eCelebV.ROOT, 'val',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif "WFLWV" in cfg.DATASET.DATASET:
        train_dataset = WFLWV_Dataset(
            cfg, cfg.WFLWV.ROOT, 'train',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_dataset = WFLWV_Dataset(
            cfg, cfg.WFLWV.ROOT, 'test',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError('Wrong Dataset')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 100.0
    last_epoch = -1

    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    map_location = f'cuda:{cfg.GPUS[0]}'
    rgb_ckpt = torch.load(cfg.MODEL.CKPT, map_location=map_location)
    rgb_state_dict = rgb_ckpt['state_dict'] if 'state_dict' in rgb_ckpt else rgb_ckpt

    model.module.load_state_dict(rgb_state_dict, strict=False)

    if 'state_dict' in rgb_state_dict:
        ckpt_keys = rgb_state_dict['state_dict'].keys()
    else:
        ckpt_keys = rgb_state_dict.keys()

    for name, param in model.module.named_parameters():
        if name in ckpt_keys:
            param.requires_grad = False # freeze
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, begin_epoch + cfg.TRAIN.NUM_EPOCH):
        train(cfg, train_loader, model, loss_function_2, optimizer, epoch,
              final_output_dir, writer_dict)
        perf_indicator = validate(
            cfg, valid_loader, model, loss_function_2, final_output_dir, writer_dict
        )

        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False
        logger.info('=> best model : {}'.format(best_model))
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

        lr_scheduler.step()

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main_function()
