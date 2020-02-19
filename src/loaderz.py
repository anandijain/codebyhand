import os
import glob

import numpy as np

import torch
import torchaudio
import torchvision
from torchvision import transforms

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import modelz

PATH = '/home/sippycups/programming/repos/usbtablet/assets/'
if __name__ == "__main__":
    
    edits = transforms.Compose([
        transforms.Resize((28, 28),),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(PATH, transform=edits)

    x, y = dataset[3]

    loader = DataLoader(dataset, batch_size=1)

    model = modelz.Net()
    model.load_state_dict(torch.load('digits.pth'))
    x = x.view(1, -1)
    yhat = model(x).view(1, -1)
    print(yhat)
    pred = yhat.max(1, keepdim=True)[1]
    print(pred)
    print(y)