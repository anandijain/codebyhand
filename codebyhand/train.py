import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms

import modelz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


BATCH_SIZE = 256
SAVE_MODEL = True

MODEL_FN = "digits.pth"


LOG_INTERVAL = 250


def prep():
    edits = transforms.Compose([transforms.ToTensor()])

    # torchvision datasets emnist currently broken
    emnist = torchvision.datasets.MNIST(
        "/home/sippycups/D/datasets/", download=False, transform=edits
    )

    data_loader = torch.utils.data.DataLoader(
        emnist, batch_size=BATCH_SIZE, shuffle=True,
    )

    x, y = emnist[0]
    print(emnist.classes)
    print(x.shape)
    print(y)

    d = {
        "data": emnist,
        "loader": data_loader,
    }
    return d


def train(d, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(d["loader"]):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data).view(-1, 10)

        # print(f'output.shape : {output.shape}')
        # print(f'data.shape : {data.shape}')
        # print(f'target: {target.item()}')

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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
        torch.save(model.state_dict(), MODEL_FN)
        print(f"model saved to {MODEL_FN}")


def test(d):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in d["loader"]:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)

            output = model(data).view(-1, 10)

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
    model = modelz.Net().to(device)
    try:
        model.load_state_dict(torch.load("digits.pth"))
    except RuntimeError:
        print("prob incompat model")

    optimizer = optim.Adam(model.parameters())
    for i in range(0, 2):
        train(d, i)
        test(d)
