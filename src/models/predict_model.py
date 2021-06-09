import argparse
import pickle

import torch
from torchvision import datasets, transforms

from src.data.data import Digits
from src.models.model import cnn_model

parser = argparse.ArgumentParser()

parser.add_argument(
    "f", help="pickle with the images to predict", type=str, required=True
)
# parser.add_argument("-t","--filetype", nargs='?', choices=('npy', 'pickle','png'),default='pickle')
args = parser.parse_args()

model = cnn_model()
cnn_model.load()
model.eval()


X, Y = pickle.load(args.f)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

testset = Digits(X, train=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, transform=transform
)

predictions = []

with torch.no_grad():
    for imgs, labels in testloader:

        n_batch = imgs.shape[0]
        y_pred = model.forward(imgs)
        _, y_pred = y_pred.topk(1, dim=1)
        predictions.extend(list(y_pred))
