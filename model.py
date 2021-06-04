import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import exists

class cnn_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(1,8,3) # 26
        self.cv2 = nn.Conv2d(8,32,3) # 13
        self.cv3 = nn.Conv2d(32,128,3) # 11

        self.max_pool1 = nn.MaxPool2d(2) # 5
        self.max_pool2 = nn.MaxPool2d(2) 

        self.fc1 = nn.Linear(3*3*128,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self,x):

        x = self.cv1(x) #26
        x = F.relu(x)
        x = self.max_pool1(x) #13

        x = self.cv2(x) # 11
        x = self.max_pool2(x) # 5
        x = F.relu(x)

        x = self.cv3(x) # 3
        x = F.relu(x)
        x = torch.flatten(x,1)

        x = self.fc1(x)
        x =F.relu(x)
        x = self.fc2(x)

        x = F.log_softmax(x,dim=1)

        return x

    def save(self):
        print("Saving model")
        torch.save(self.state_dict(), 'saves/checkpoint.pth')

    def load(self):
        print("Loading model")
        state_dict = torch.load('saves/checkpoint.pth')
        self.load_state_dict(state_dict)

    def save_best_accuracy(acc):
        with open("best_accuracy.txt",'w') as handle:
            handle.writelines(str(acc))

    def get_best_accuracy():

        if not exists("best_accuracy.txt"):
            return 0
        with open("best_accuracy.txt",'r') as handle:
            acc = float(handle.readlines()[0])
            return acc