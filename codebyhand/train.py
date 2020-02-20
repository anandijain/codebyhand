import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms

from codebyhand import loaderz
from codebyhand import modelz
from codebyhand import utilz
from codebyhand import macroz as mz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

BATCH_SIZE = 128

MODEL_FN = f"{mz.SRC_PATH}convemnist2.pth"

LOAD_MODEL = True
SAVE_MODEL = True

LOG_INTERVAL = 250


def get_mnist():
    # torchvision datasets emnist currently broken, copy master torchvision mnist.py to local install to fix
    emnist = torchvision.datasets.EMNIST(
        mz.DATA_DIR, split="byclass", download=True, transform=loaderz.TO_MNIST
    )
    emnist.classes = sorted(emnist.classes)
    return emnist


def prep(data, verbose=True):

    num_classes = len(data.classes)

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True,
    )

    model = modelz.ConvNet(out_dim=num_classes).to(device)

    if LOAD_MODEL:
        try:
            model.load_state_dict(torch.load(MODEL_FN))
            print(f"loaded model")
        except RuntimeError:
            print("prob incompat model")
        except:
            print("cant load, other reason")

    optimizer = optim.Adadelta(model.parameters())

    d = {"data": data, "loader": data_loader,
         "model": model, "optimizer": optimizer}
    if verbose:
        print(d)
    return d


def train_epoch(d, epoch, save_model=SAVE_MODEL, model_fn=MODEL_FN, return_preds=False):
    d["model"].train()
    losses = []
    preds = []
    for batch_idx, (data, target) in enumerate(d["loader"]):
        data, target = data.to(device), target.to(device)

        d["optimizer"].zero_grad()
        output = d["model"](data, use_dropout=True)  # .view(-1, 10)

        loss = F.nll_loss(output, target)
        loss.backward()
        d["optimizer"].step()

        if return_preds:
            preds.append(output)
            losses.append(loss.item())

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
        torch.save(d["model"].state_dict(), model_fn)
        print(f"model saved to {model_fn}")

    if return_preds:
        return preds, losses


def test_epoch(d):
    with torch.no_grad():
        d["model"].eval()
        test_loss = 0
        correct = 0
        for data, target in d["loader"]:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)

            output = d["model"](data).view(-1, 10)

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
    d = prep(data=get_mnist())
    for i in range(0, 1):
        train_epoch(d, i)
        # test(d)
