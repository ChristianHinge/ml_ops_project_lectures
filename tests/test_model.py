import os
from src.models import main, model
import torch
import numpy as np
import pytest

dirname = os.path.dirname(__file__)

def test_evaluate():
    main.TrainOREvaluate().load_and_eval(log_wandb=False)

@pytest.mark.parametrize("shape",[(2,1,28,28),(1,1,28,28)])
def test_model_output(shape):
    model1 = model.cnn_model()
    imgs = torch.zeros(shape)
    y_pred = model1.forward(imgs)
    assert y_pred.shape == (shape[0],10)


def test_model_input():
    model1 = model.cnn_model()
    with pytest.raises(TypeError):
        model1.forward(np.zeros(3,1,28,28))
    
    with pytest.raises(ValueError):
        model1.forward(torch.zeros(2,1,28,28,0))
    
    with pytest.raises(ValueError):
        model1.forward(torch.zeros(2,1,29,28))

