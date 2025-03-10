import time
import logging
import os

import numpy as np
import torch

from torch.nn import functional as F
from utils import AverageMeter

import wandb
logger = logging.getLogger(__name__)


def train(config, train_loader, model, loss_function, optimizer, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    NME_stage1 = AverageMeter()
    NME_stage2 = AverageMeter()
    NME_stage3 = AverageMeter()
    loss_average = AverageMeter()

    model.train()


    end = time.time()

    for i, (input, event_input, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        event_input = event_input.cuda(non_blocking=True)
        ground_truth = meta['Points'].cuda().float()
        landmarks_ev = model(input, event_input)    # attn : (6, 32, 98, 98)

        R_loss_1 = loss_function(landmarks_ev[0], ground_truth)
        R_loss_2 = loss_function(landmarks_ev[1], ground_truth)
        R_loss_3 = loss_function(landmarks_ev[2], ground_truth)
        loss = 0.2 * R_loss_1 + 0.3 * R_loss_2 + 0.5 * R_loss_3

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        NME_stage1.update(loss.item(), input.size(0))
        NME_stage2.update(R_loss_2.item(), input.size(0))
        NME_stage3.update(R_loss_3.item(), input.size(0))

        loss_average.update(loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if writer_dict is not None:
            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'NME_stage1 {NME_stage1.val:.5f} ({NME_stage1.avg:.5f})\t' \
                      'NME_stage2 {NME_stage2.val:.5f} ({NME_stage2.avg:.5f})\t' \
                      'NME_stage3 {NME_stage3.val:.5f} ({NME_stage3.avg:.5f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time, loss=loss_average, NME_stage1=NME_stage1,
                    NME_stage2=NME_stage2, NME_stage3=NME_stage3)
                logger.info(msg)

                wandb.log({
                    "epoch": epoch,
                    "batch": i,
                    "train_loss": loss_average.val,
                    "NME_stage1": NME_stage1.val,
                    "NME_stage2": NME_stage2.val,
                    "NME_stage3": NME_stage3.val,
                    "batch_time/val": batch_time.val,
                    "data_time/val": data_time.val
                }, step=writer_dict['train_global_steps'])  # step으로 global_steps를 사용

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', loss_average.val, global_steps)
                writer.add_scalar('NME1', NME_stage1.val, global_steps)
                writer.add_scalar('NME2', NME_stage2.val, global_steps)
                writer.add_scalar('NME3', NME_stage3.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
