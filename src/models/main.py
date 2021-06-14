import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from src.models.model import cnn_model
import torchvision
from src.data.data import mnist

import dotenv
import os


class TrainOREvaluate:

    def __init__(self):
        self.project_dir = os.path.dirname(__file__)
        #dotenv_path = os.path.join(project_dir, '.env')
        #dotenv.load_dotenv(dotenv_path)

        self.model = cnn_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.NLLLoss()

    def train_epoch(self,test_mode=False):

        trainloader, _ = mnist()
        self.model.train()

        n_images = 0
        total_loss = 0
        for imgs, labels in trainloader:

            n_batch = imgs.shape[0]
            self.optimizer.zero_grad()
            y_pred = self.model.forward(imgs)
            loss = self.criterion(y_pred, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * n_batch
            n_images += n_batch
            if test_mode and n_images > 250:
                break

        return total_loss / n_images

        
    def eval_model(self,log_wandb=True):

        _, testloader = mnist()
        self.model.eval()

        n_images = 0
        n_correct = 0

        with torch.no_grad():
            for imgs, labels in testloader:

                n_batch = imgs.shape[0]
                y_pred = self.model.forward(imgs)
                _, y_pred = y_pred.topk(1, dim=1)
                n_correct += (y_pred == labels.view(*y_pred.shape)).sum()
                n_images += n_batch

        print(f"Eval accuracy: {n_correct/n_images*100:.2f}%")
        #torchvision.utils.make_grid(img)
        if log_wandb:
            captions = [f"True: {t.item()}, Pred: {p.item()}" for t,p in zip(labels,y_pred)]
            wandb.log({"Final predictions":[wandb.Image(imgs[i,],caption=captions[i]) for i in range(imgs.shape[0])]})

    def train_epochs(self,epochs=5,log_wandb=True):
        loss = np.zeros(epochs)
        for i in range(epochs):
            loss[i] = self.train_epoch()
            print(f"Epoch {i+1}/{epochs}. Loss: {loss[i].round(3)}")
            
            if log_wandb:
                wandb.log({"CrossEntroyLoss":loss[i]},step=i+1)
        
        f = plt.figure()
        plt.plot(np.arange(1, epochs + 1), loss)
        plt.title("Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.project_dir,"../../reports/figures/train_loss.png"))


    def train_and_save(self,log_wandb=True):
        print("Training mode")
        self.train_epochs(5,log_wandb=log_wandb)
        self.model.save()


    def load_and_eval(self,log_wandb=True):
        print("Evaluation mode")
        self.model.load()
        self.eval_model(log_wandb=log_wandb)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="train or eval", nargs="?", choices=("train", "eval"))
    
    args = parser.parse_args()
    
    if args.mode == "train":
        TrainOREvaluate().train_and_save()
    elif args.mode == "eval":
        TrainOREvaluate().load_and_eval()


    