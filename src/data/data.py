import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def mnist():
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    """
  # Download and load the training data
  trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

  # Download and load the test data
  testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
  """

    # Download and load the training data
    trainset = datasets.MNIST(
        "../../data/raw/MNIST/", download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST(
        "../../data/raw/MNIST/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader


class Digits(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y=None, transform=None, train=True):

        self.transform = transform
        self.X = X
        self.Y = Y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.X[idx]

        if self.transform:
            X = self.transform(X)

        if self.train:
            Y = self.Y[idx]
            return X, Y

        return X
