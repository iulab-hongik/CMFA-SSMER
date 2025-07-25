import argparse

from Config import cfg
from Config import update_config

from utils import create_logger
from model import Sparse_alignment_network
from Dataloader import WFLWV_Dataset, eCelebV_Dataset, ESIE_Dataset
from utils import AverageMeter
from utils.save import save_img, save_comparison


from tensorboardX import SummaryWriter

import torch
import cv2
import numpy as np
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

def calcuate_loss(name, pred, gt, trans, thres=0.10):

    pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

    if name == 'WFLWV' or 'CelebV' in name:
        norm = np.linalg.norm(gt[60, :] - gt[72, :])
    elif name in ['ESIE']:
        pred = pred[[54, 76, 82, 96, 97], :]
        norm = np.linalg.norm(gt[3, :] - gt[4, :])  # inter-pupil
    else:
        raise ValueError('Wrong Dataset')

    errors = np.linalg.norm(pred - gt, axis=1) / norm
    nme = np.mean(errors)
    fr = np.mean(errors > thres)

    error_range = np.linspace(0, thres, num=100)
    ced_curve = [np.mean(errors <= e) for e in error_range]
    auc = np.trapz(ced_curve, error_range) / thres

    return nme, fr, auc

def main_function():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg.TARGET)
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if "eCelebV" in cfg.DATASET.DATASET:
        model = Sparse_alignment_network(cfg.eCelebV.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.eCelebV.INITIAL_PATH, cfg)
    elif cfg.DATASET.DATASET == "ESIE":
        model = Sparse_alignment_network(cfg.ESIE.NUM_POINT, cfg.MODEL.OUT_DIM,
                                         cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                         cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                         cfg.TRANSFORMER.FEED_DIM, cfg.ESIE.INITIAL_PATH, cfg)
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


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if "eCelebV" in cfg.DATASET.DATASET:
        valid_dataset = eCelebV_Dataset(
            cfg, cfg.eCelebV.ROOT, 'test',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif "ESIE" in cfg.DATASET.DATASET:
        valid_dataset = ESIE_Dataset(
            cfg, cfg.ESIE.ROOT, 'test',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif "WFLWV" in cfg.DATASET.DATASET:
        valid_dataset = WFLWV_Dataset(
            cfg, cfg.WFLWV.ROOT, 'test',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError('Wrong Dataset')

    # 验证数据迭代器
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = cfg.TEST.CHECKPOINT
    checkpoint = torch.load(checkpoint_file)

    model.module.load_state_dict(checkpoint)
    logger.info(checkpoint_file)

    nme_list = AverageMeter()
    fr_list = AverageMeter()
    auc_list = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, event_input, meta) in enumerate(valid_loader):
            Annotated_Points = meta['Annotated_Points'].numpy()[0]
            Trans = meta['trans'].numpy()[0]
            outputs_initial = model(input.cuda(), event_input.cuda())

            stage1 = outputs_initial[0][0, -1, :, :].cpu().numpy()
            stage2 = outputs_initial[1][0, -1, :, :].cpu().numpy()
            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            stage1 = stage1 * cfg.MODEL.IMG_SIZE
            stage2 = stage2 * cfg.MODEL.IMG_SIZE
            stage3 = output * cfg.MODEL.IMG_SIZE

            stage1 = (stage1 - Trans[:, 2]) @ np.linalg.inv(Trans[:, 0:2].T)
            stage2 = (stage2 - Trans[:, 2]) @ np.linalg.inv(Trans[:, 0:2].T)
            stage3 = (stage3 - Trans[:, 2]) @ np.linalg.inv(Trans[:, 0:2].T)
            if cfg.DATASET.DATASET in ['ESIE']:
                stage1 = stage1[[54, 76, 82, 96, 97], :]
                stage2 = stage2[[54, 76, 82, 96, 97], :]
                stage3 = stage3[[54, 76, 82, 96, 97], :]
            # save_img(meta, stage1)
            # save_comparison(meta, stage1, stage2, stage3, gt=Annotated_Points)

            nme, fr, auc = calcuate_loss(cfg.DATASET.DATASET, output * cfg.MODEL.IMG_SIZE, Annotated_Points, Trans)
            nme_list.update(nme, input.size(0))
            fr_list.update(fr, input.size(0))
            auc_list.update(auc, input.size(0))

            msg = 'Epoch: [{0}/{1}]\t' \
                  'NME: {nme:.3f}%\t' \
                  'FR: {fr:.3f}%\t' \
                  'AUC: {auc:.3f}\t'.format(
                i, len(valid_loader), nme=nme_list.avg * 100.0, fr=fr_list.avg * 100.0, auc=auc_list.avg)

            logger.info(msg)
    logger.info("=====Finish=====")

if __name__ == '__main__':
    main_function()

