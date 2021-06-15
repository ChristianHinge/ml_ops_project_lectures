from my_mnist.data import data
from torchvision import transforms
from my_mnist.models.model import cnn_model
import torch
import pytest

def test_mnist():
    train_loader, test_loader = data.mnist()
    assert tuple(train_loader.dataset.data.shape) == (60000,28,28)
    assert tuple(test_loader.dataset.data.shape) == (10000,28,28)
    assert tuple(train_loader.dataset.targets.unique()) == tuple(i for i in range(10))
    assert tuple(test_loader.dataset.targets.unique()) == tuple(i for i in range(10))

@pytest.mark.parametrize("train",[True,False])
def test_digits(train):

    # Data loading
    X_train = torch.ones(100,1,28,28)
    Y_train = torch.cat([torch.arange(10)]*10,dim=0)

    class MultTrns:
        def __init__(self,scale):
            self.scale = scale
        def __call__(self,X):
            return X*self.scale
    
    transform = transforms.Compose(
        [
            MultTrns(2),
        ]
    )

    train_dataset = data.Digits(X=X_train,Y=Y_train, transform=transform, train=train)
    train_loader =  torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    assert tuple(train_loader.dataset.X.shape) == (100,1,28,28)
    assert tuple(train_loader.dataset.Y.unique()) == tuple(i for i in range(10))

    x,y = iter(train_loader).__next__()
    assert x.shape == (64,1,28,28)
    assert y.shape == (64,)
    assert (x==2).all()

    pred = cnn_model().forward(x)
        