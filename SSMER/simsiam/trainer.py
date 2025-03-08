import torch

import time
import math
import shutil
import logging


def dual_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    fail_file_path = 'fail.txt'
    with open(fail_file_path, 'w') as fail_file:
        for i, (image1, image2) in enumerate(train_loader):  # 이미지 경로를 함께 가져옴
            try:
                # measure data loading time
                data_time.update(time.time() - end)

                if args.gpu is not None:
                    image1 = image1.cuda(args.gpu, non_blocking=True)
                    image2 = image2.cuda(args.gpu, non_blocking=True)

                # compute output and loss
                p1, p2, z1, z2 = model(x1=image1, x2=image2)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

                losses.update(loss.item(), image1.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except Exception as e:
                # 오류 발생 시 이미지 경로를 fail.txt에 기록
                fail_file.write(f'Error processing image: {e}\n')  # 경로 기록
                print(e)
                continue  # 오류가 발생한 이미지는 건너뜁니다.

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    logging.info(f'Epoch: {epoch+1}\tLosses: {losses}')


def trio_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    fail_file_path = 'fail.txt'
    with open(fail_file_path, 'w') as fail_file:
        for i, (image1, image2, image3) in enumerate(train_loader):  # 이미지 경로를 함께 가져옴
            try:
                # measure data loading time
                data_time.update(time.time() - end)

                if args.gpu is not None:
                    image1 = image1.cuda(args.gpu, non_blocking=True)
                    image2 = image2.cuda(args.gpu, non_blocking=True)
                    image3 = image3.cuda(args.gpu, non_blocking=True)

                # compute output and loss
                outputs = [
                    model(x1=image1, x2=image2),
                    model(x1=image2, x2=image3),
                    model(x1=image1, x2=image3)
                ]
                cosine_losses = []

                for (p1, p2, z1, z2) in outputs:
                    # Cosine Loss
                    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                    cosine_losses.append(loss)
                total_loss = sum(cosine_losses)
                losses.update(total_loss.item(), image1.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            except Exception as e:
                # 오류 발생 시 이미지 경로를 fail.txt에 기록
                fail_file.write(f'Error processing image: {e}\n')  # 경로 기록
                print(e)
                continue  # 오류가 발생한 이미지는 건너뜁니다.

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    logging.info(f'Epoch: {epoch+1}\tLosses: {losses}')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def calculate_losses(criterion_cosine, criterion_kl, outputs):
    negative_cosine_losses = []
    jsd_losses = []

    for (p1, p2, z1, z2) in outputs:
        # Cosine Loss
        negative_cosine_loss = -(criterion_cosine(p1, z2).mean() + criterion_cosine(p2, z1).mean()) * 0.5
        negative_cosine_losses.append(negative_cosine_loss)

        # JSD Loss
        jsd_loss = 0.5 * (criterion_kl(p1.log_softmax(dim=1), p2.softmax(dim=1)) +
                          criterion_kl(p2.log_softmax(dim=1), p1.softmax(dim=1)))
        jsd_losses.append(jsd_loss)

    # Sum up all individual losses
    total_negative_cosine_loss = sum(negative_cosine_losses)
    total_jsd_loss = sum(jsd_losses)

    return total_negative_cosine_loss + total_jsd_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
