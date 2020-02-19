import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms

from codebyhand import modelz
from codebyhand import macroz as mz
from codebyhand import loaderz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

BATCH_SIZE = 32

MODEL_FN = f"{mz.SRC_PATH}convemnist.pth"
LOAD_MODEL = True
SAVE_MODEL = True


LOG_INTERVAL = 250


def prep(verbose=True):

    # torchvision datasets emnist currently broken, copy master torchvision mnist.py to local install to fix
    emnist = torchvision.datasets.EMNIST(
        "/home/sippycups/D/datasets/", split='byclass', download=False, transform=loaderz.TO_MNIST
    )
    print(emnist.classes)
    emnist.classes = sorted(emnist.classes)
    print(emnist.classes)
    num_classes = len(emnist.classes)
    data_loader = torch.utils.data.DataLoader(
        emnist, batch_size=BATCH_SIZE, shuffle=False,
    )

    model = modelz.ConvNet(out_dim=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_FN))
        print(f'loaded model')
    except RuntimeError:
        print("prob incompat model")
    except:
        print('cant load, other reason')

    
    optimizer = optim.Adam(model.parameters())
    x, y = emnist[0]

    d = {
        "data": emnist,
        "loader": data_loader,
        'model': model,
        'optimizer': optimizer
    }
    if verbose:
        print(d)
    return d


def train(d, epoch):
    d['model'].train()
    for batch_idx, (data, target) in enumerate(d["loader"]):
        data, target = data.to(device), target.to(device)
        # data = data.view(-1, 784)
        d['optimizer'].zero_grad()
        output = d['model'](data)# .view(-1, 10)

        loss = F.nll_loss(output, target)
        loss.backward()
        d['optimizer'].step()
        if batch_idx % LOG_INTERVAL == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(d["loader"].dataset),
                    100.0 * batch_idx / len(d["loader"]),
                    loss.item(),
                )
            )

    if SAVE_MODEL:
        torch.save(d['model'].state_dict(), f'{MODEL_FN}')
        print(f"model saved to {MODEL_FN}")


def test(d):
    with torch.no_grad():
        d['model'].eval()
        test_loss = 0
        correct = 0
        for data, target in d["loader"]:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)

            output = d['model'](data).view(-1, 10)

            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(d["loader"].dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(d["loader"].dataset),
                100.0 * correct / len(d["loader"].dataset),
            )
        )


if __name__ == "__main__":
    d = prep()
    for i in range(0, 2):
        train(d, i)
        # test(d)
