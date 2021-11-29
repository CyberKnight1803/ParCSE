import os
import torch

EPOCHS = 3
PATH_DATASETS = os.environ.get("PATH_DATASETS", "./datasets")
PATH_BASE_MODELS = os.environ.get("PATH_BASE_MODELS", "./base_models")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128
NUM_WORKERS = max(int(os.cpu_count() / 2), os.cpu_count() - 2)
