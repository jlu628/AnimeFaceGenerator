import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = 0.5
STD = 0.5
BATCH_SIZE = 16
LATENT_SIZE = (512, 1, 1)
EPOCHS = 100
LOSS_FN = torch.nn.functional.binary_cross_entropy
OPTIMIZER = torch.optim.Adam
BETAS = (0.5, 0.999)
D_LR = 0.0002
D_LR_DECAY = 1e-6
G_LR = 0.0002
G_LR_DECAY = 1e-6
CORES = 4
LOAD_MODELS = False