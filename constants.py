import torch

EPOCHS = 5
BATCH_SIZE = 32
N_EVAL = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")