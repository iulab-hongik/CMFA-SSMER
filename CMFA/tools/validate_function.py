import numpy as np
import time
import torch
import logging
import os

from utils import AverageMeter
from backbone import Backbone
import wandb

logger = logging.getLogger(__name__)

def validate(config, val_loader, model_ev, loss_function, output_dir, writer_dict=None,
             edge_criterion=None):
    batch_time = AverageMeter()
    loss_average = AverageMeter()
    NME_stage1 = AverageMeter()
    NME_stage2 = AverageMeter()
    NME_stage3 = AverageMeter()

    model_ev.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, event_input, meta) in enumerate(val_loader):
            landmarks_ev = model_ev(input, event_input)
            ground_truth = meta['Points'].cuda().float()

            R_loss1 = loss_function(landmarks_ev[0][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            R_loss2 = loss_function(landmarks_ev[1][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            R_loss3 = loss_function(landmarks_ev[2][:, config.TRANSFORMER.NUM_DECODER-1:config.TRANSFORMER.NUM_DECODER, :, :].detach(),
                                    ground_truth)
            NME_stage1.update(R_loss1.item(), event_input.size(0))
            NME_stage2.update(R_loss2.item(), event_input.size(0))
            NME_stage3.update(R_loss3.item(), event_input.size(0))

            loss = 0.2 * R_loss1 + 0.3 * R_loss2 + 0.5 * R_loss3

            loss_average.update(loss.item(), event_input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if writer_dict is not None:
                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                          'NME_stage1 {NME_stage1.val:.5f} ({NME_stage1.avg:.5f})\t' \
                          'NME_stage2 {NME_stage2.val:.5f} ({NME_stage2.avg:.5f})\t' \
                          'NME_stage3 {NME_stage3.val:.5f} ({NME_stage3.avg:.5f})\t'.format(
                        i, len(val_loader), batch_time=batch_time,
                        speed=input.size(0) / batch_time.val,
                        loss=loss_average, NME_stage1=NME_stage1, NME_stage2=NME_stage2,
                        NME_stage3=NME_stage3)
                    logger.info(msg)

                    wandb.log({
                        "validation/loss": loss_average.val,
                        "validation/NME_stage1": NME_stage1.val,
                        "validation/NME_stage2": NME_stage2.val,
                        "validation/NME_stage3": NME_stage3.val
                    }, step=writer_dict['train_global_steps'])

                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('validate_loss', loss_average.val, global_steps)
                    writer.add_scalar('validate_NME1', NME_stage1.val, global_steps)
                    writer.add_scalar('validate_NME2', NME_stage2.val, global_steps)
                    writer.add_scalar('validate_NME3', NME_stage3.val, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

        if writer_dict is not None:
            msg = 'Stage1: ({NME_stage1.avg:.5f})\t' \
                  'Stage2: ({NME_stage2.avg:.5f})\t' \
                  'Stage3: ({NME_stage3.avg:.5f})\t'.format(
                NME_stage1=NME_stage1, NME_stage2=NME_stage2, NME_stage3=NME_stage3)
            logger.info(msg)

            wandb.log({
                "validation/NME_stage1_avg": NME_stage1.avg,
                "validation/NME_stage2_avg": NME_stage2.avg,
                "validation/NME_stage3_avg": NME_stage3.avg
            }, step=writer_dict['train_global_steps'])

        return NME_stage3.avg
