import os
import logging
import time
from pathlib import Path
import torch.distributed as dist

def create_logger(cfg, phase='train'):
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    root_output_dir = Path(cfg.OUTPUT_DIR)
    if rank == 0 and not root_output_dir.exists():
        print(f'=> crate {root_output_dir}')
        root_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    final_output_dir = root_output_dir / dataset / model
    if rank == 0 and not final_output_dir.exists():
        print(f'=> crate {final_output_dir}')
        final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{model}_{phase}_{time_str}.log'
    final_log_file = final_output_dir / log_file

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if rank == 0 and not logger.hasHandlers():
        file_handler = logging.FileHandler(final_log_file)
        file_formatter = logging.Formatter('%(asctime)-15s %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)

        print("=> Logger initialized for Rank 0")

    if rank != 0 and not logger.hasHandlers():
        logger.addHandler(logging.NullHandler())

    tensorboard_log_dir = None
    if rank == 0:
        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / (phase + '_' + time_str)
        if not tensorboard_log_dir.exists():
            print(f'=> creating {tensorboard_log_dir}')
            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir) if tensorboard_log_dir else ''
