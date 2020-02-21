import os
import glob

import numpy as np
import torch

from PIL import Image, ImageTk

from codebyhand import loaderz
from codebyhand import macroz as mz


def migrate_files():
    classed_fns = []
    root = '/home/sippycups/codebyhand/assets/EMNIST/'
    for c in mz.EMNIST_CLASSES:
        files = glob.glob(f'{mz.IMGS_PATH}**_{c}.*')
        [os.rename(f, f'{root}{c}/{f.split("/")[-1]}') for f in files]
        classed_fns.append(files)
    return classed_fns


def char_from_fn(fn:str)->str:
    return fn.split('_')[-1].split('.')[0]


def prep_personal_dataset(dataset, dataset_name:str):
    # only run once to set up
    dataset_root = f'{mz.ASSETS_PATH}{dataset_name}'
    os.mkdir(dataset_root)
    dirnames = [dataset_root + c for c in dataset.classes]
    os.makedirs(dirnames)


def np_to_emnist_tensor(char):
    char = Image.fromarray(char)
    return loaderz.TO_MNIST(char)


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
    return img[y0 - pad : y1 + pad, x0 - pad : x1 + pad]


def get_chars(img, bounds):
    return [crop_char(img, b) for b in bounds]


def save_char(char: np.array, fn: str):
    shape = char.shape
    im = Image.fromarray(char)

    im.save(f"{mz.IMGS_PATH}{fn}.png")
    return im

def save_labeled_char(char: np.array, label:str, fn: str, verbose=False):
    shape = char.shape
    im = Image.fromarray(char)
    save_fn = f"{mz.MYEMNIST}{label}/{fn}.png"
    im.save(save_fn)
    if verbose:
        print(f'saved to {save_fn}')
    return im


def infer_char(model, char: Image, device):
    x = loaderz.TO_MNIST(char).to(device)

    with torch.no_grad():
        output = model(x[None, ...])

    return emnist_val(output)
