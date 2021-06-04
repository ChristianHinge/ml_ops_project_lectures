import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import argparse
from data import trainloader, testloader
import numpy as np
from model import cnn_model

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='train or eval', nargs='?', choices=('train', 'eval'))

args = parser.parse_args()


def train_epoch():

    model.train()

    n_images = 0 
    total_loss = 0
    for imgs, labels in trainloader:
        
        n_batch = imgs.shape[0]
        optimizer.zero_grad()
        y_pred = model.forward(imgs)
        loss = criterion(y_pred,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*n_batch
        n_images += n_batch


    return total_loss/n_images


def eval():

    model.eval()

    n_images = 0 
    n_correct = 0

    with torch.no_grad():
        for imgs, labels in testloader:
            
            n_batch = imgs.shape[0]
            y_pred = model.forward(imgs)
            _,y_pred = y_pred.topk(1,dim=1)
            n_correct += (y_pred == labels.view(*y_pred.shape)).sum()
            n_images += n_batch

    print(f"Eval accuracy: {n_correct/n_images*100:.2f}%")

def train_epochs(epochs=5):
    loss = np.zeros(epochs)
    for i in range(epochs):
        loss[i] = train_epoch()
        print(f"Epoch {i+1}/{epochs}. Loss: {loss[i].round(3)}")
    
    plt.plot(np.arange(1,epochs+1),loss)
    plt.title("Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")



if args.mode == "train":
    print("Training mode")
    model = cnn_model()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.NLLLoss()
    train_epochs(2)
    model.save()

elif args.mode == "eval":
    print("Evaluation mode")
    model = cnn_model()
    model.load()
    eval()
