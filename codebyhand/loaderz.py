import os
import glob

import numpy as np

import torch
import torchaudio
import torchvision
from torchvision import transforms

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from codebyhand import modelz
from codebyhand import loaderz
from codebyhand import macroz as mz


TO_MNIST = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

class Chars(Dataset):
    def __init__(self, data:list):
        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":

    print(mz.IMGFOLDER_PATH)
    dataset = torchvision.datasets.ImageFolder(
        mz.IMGFOLDER_PATH, transform=TO_MNIST)
    x, y = dataset[3]

    loader = DataLoader(dataset, batch_size=1)

    model = modelz.ConvNet()
    model.load_state_dict(torch.load(f"{mz.SRC_PATH}digits.pth"))
    preds = {}
    for i, (x, y) in enumerate(dataset):
        x = x.view(1, -1)
        yhat = model(x).view(1, -1)
        pred = yhat.max(1, keepdim=True)[1]
        preds[i] = pred
        # print(f'yhat {yhat}')

    print(preds)
