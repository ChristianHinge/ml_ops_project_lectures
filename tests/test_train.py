from torch.optim import optimizer
from src.models.main import TrainOREvaluate
import torch

def test_trainloss_decrease():
    toe = TrainOREvaluate()

    torch.manual_seed(0)
    toe.optimizer = torch.optim.Adam(toe.model.parameters(),lr=0)
    loss_before = toe.train_epoch(test_mode=True)

    torch.manual_seed(0)
    toe.optimizer = torch.optim.Adam(toe.model.parameters(),lr=1e-3)
    loss_after = toe.train_epoch(test_mode=True)

    assert loss_before > loss_after