import os
import torch

EPOCHS = 50
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 4
NUM_WORKERS = int(os.cpu_count() / 2)
