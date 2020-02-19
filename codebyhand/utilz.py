import numpy as np
import torch

from PIL import Image, ImageTk

from codebyhand import loaderz
from codebyhand import macroz as mz


def np_to_emnist_tensor(char):
    char = Image.fromarray(char)
    return loaderz.TO_MNIST(char)[None, ...]


def emnist_val(yhat):
    pred_idx = yhat.max(1, keepdim=True)[1]
    pred = mz.EMNIST_CLASSES[pred_idx]
    return pred


def norm_stroke(s: np.ndarray) -> np.ndarray:
    return s - s[0]


def stroke_bounds(s: np.ndarray) -> np.ndarray:
    xmin = s[:, 0].min()
    xmax = s[:, 0].max()

    ymin = s[:, 1].min()
    ymax = s[:, 1].max()
    return [xmin, xmax, ymin, ymax]


def crop_char(img: np.array, b: list, pad: int = 5, scale: float = 0.75):
    x0, x1, y0, y1 = [int(scale * coord) for coord in b]
    return img[y0 - pad: y1 + pad, x0 - pad: x1 + pad]


def get_chars(img, bounds):
    return [crop_char(img, b) for b in bounds]


def save_char(char: np.array, fn: str):
    shape = char.shape
    im = Image.fromarray(char)

    im.save(f"{mz.IMGS_PATH}{fn}.png")
    return im


def infer_char(model, char: Image):
    x = loaderz.TO_MNIST(char)

    with torch.no_grad():
        output = model(x[None, ...])

    return emnist_val(output)

