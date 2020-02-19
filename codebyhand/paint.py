"""
todo single pen click throws index error

cursive wont work for the bounding box grabber 
https://stackoverflow.com/questions/41940945/saving-canvas-from-tkinter-to-file

"""

from tkinter import *
from tkinter.colorchooser import askcolor

import io
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from PIL import Image, ImageTk

import codebyhand as cbh

from codebyhand import modelz
from codebyhand import loaderz
from codebyhand import macroz as mz

MODEL_FN = f'{mz.SRC_PATH}convemnist2.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

config = {"width": 1400, "height": 500, "pen_radius": 5, "bg": "white"}


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = "black"

    def __init__(self):
        self.root = Tk()

        self.state = []
        self.state_dict = {}
        self.cur_stroke = []
        self.img = None
        self.state_bounds = []
        self.chars = []
        self.live_infer = False
        self.auto_erase = False

        self.model = modelz.ConvNet(out_dim=62)
        self.model.load_state_dict(torch.load(MODEL_FN))

        self.optimizer = optim.Adadelta(self.model.parameters())

        self.c = gen_canvas(self.root)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = config["pen_radius"]
        self.color = self.DEFAULT_COLOR
        self.active_button = self.live_infer_button
        self.c.bind("<B1-Motion>", self.paint)
        self.c.bind("<ButtonRelease-1>", self.reset)

    def live_infer_toggle(self):
        if self.live_infer == True:
            self.live_infer = False
        else:
            self.live_infer = True

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def toggle_auto_erase(self):
        if self.auto_erase == True:
            self.auto_erase = False
        else:
            self.auto_erase = True

    def use_eraser(self):
        self.c.delete("all")

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=self.line_width,
                fill=self.color,
                capstyle=ROUND,
                smooth=TRUE,
                splinesteps=36,
            )

        self.old_x = event.x
        self.old_y = event.y
        self.cur_stroke.append([self.old_x, self.old_y])

    def reset(self, event):
        self.old_x, self.old_y = None, None
        s = np.array(self.cur_stroke)
        self.state.append(s)
        self.state_bounds.append(stroke_bounds(s))
        self.cur_stroke = []
        if self.live_infer:
            self.save()
        if self.auto_erase:
            self.clear()

    def clear(self):
        self.chars = []
        self.state_bounds = []
        self.state = []
        self.img = None
        self.c.delete("all")

    def save(self):
        self.ps = self.c.postscript(colormode="gray")
        self.img = save_canvas(self.ps, save=True)
        self.chars = get_chars(self.img, self.state_bounds)
        self.pil_chars = [save_char(char, str(i))
                          for i, char in enumerate(self.chars)]
        self.infer()

    def info(self):
        print(f"state: {self.state}")
        print(f"chars: {self.chars}")

        print(f"state_bounds: {self.state_bounds}")
        print(f"num_strokes: {len(self.state)}")
        if len(self.chars) > 0:
            print(f"img shape: {self.img.shape}")

    def infer(self):
        pred_str = []
        for i, char in enumerate(self.pil_chars):
            x = loaderz.TO_MNIST(char)

            with torch.no_grad():
                yhat = self.model(x[None, ...])  # .view(1, -1))

            pred = emnist_val(yhat)
            pred_str.append(pred)

        print(f'pred_str: {pred_str}\n')

    def live_train(self):
        print('type in what you wrote')
        x = input()
        if len(x) != len(self.chars):
            print(f'len(x): {len(x)}, len(chars): {len(self.chars)}')
            print('multi stroke chars not supported yet')
            return
        npclasses = np.array(mz.EMNIST_CLASSES)
        target_idxs = [np.where(npclasses == elt)[0] for elt in x]

        # target_idxss = np.where(npclasses == x)

        print(target_idxs)
        # print(target_idxss)

        for char, target_idx in zip(self.chars, target_idxs):
            char = Image.fromarray(char)
            char = loaderz.TO_MNIST(char)
            print(f'target_idxa{target_idx}')
            target_idx = target_idx[0]
            print(f'target_idxb{target_idx}')
            
            y_char = mz.EMNIST_CLASSES[target_idx]

            self.optimizer.zero_grad()
            print(char.shape)
            yhat = self.model(char[None, ...])
            pred_char = emnist_val(yhat)
            print(f'pred_char:{pred_char}, target{y_char}')
            print(f'pred_char:{yhat}, target{yhat.shape}')
            
            target = torch.tensor(target_idx).view(-1)

            loss = F.nll_loss(yhat, target)
            loss.backward()

            self.optimizer.step()

        torch.save(self.model.state_dict(), MODEL_FN)
        print(f"model saved to {MODEL_FN}")

def gen_button(root, text:str, fxn, column:int):
    b = Button(
        root, text=text, command=fxn)
    b.grid(row=0, column=0)
    return b

def gen_canvas(root):


    infer_button = Button(root, text="infer", command=infer)
    infer_button.grid(row=0, column=1)

    info_button = Button(root, text="info", command=info)
    info_button.grid(row=0, column=2)

    auto_erase_button = Button(
        root, text="auto_erase", command=toggle_auto_erase)
    auto_erase_button.grid(row=0, column=4)

    save_button = Button(root, text="save", command=save)
    save_button.grid(row=0, column=3)

    train_button = Button(
        root, text="live_train", command=live_train)
    train_button.grid(row=0, column=5)

    c = Canvas(
        root, bg=config["bg"], width=config['width'], height=config['height'])
    c.grid(
        row=1, columnspan=6,
    )
    return c

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


def save_canvas(ps, fn="test", save=False):
    img = Image.open(io.BytesIO(ps.encode("utf-8")))

    if save:
        img.save(f"{mz.IMGS_PATH}{fn}.jpg", "jpeg")
    return np.asarray(img)


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


if __name__ == "__main__":
    x = Paint()
