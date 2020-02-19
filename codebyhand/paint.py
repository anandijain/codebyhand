"""
todo single pen click throws index error

cursive wont work for the bounding box grabber 
https://stackoverflow.com/questions/41940945/saving-canvas-from-tkinter-to-file

"""
import io
import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from tkinter import *
from tkinter.colorchooser import askcolor

from PIL import Image, ImageTk

from codebyhand import macroz as mz

from codebyhand import modelz
from codebyhand import loaderz
from codebyhand import utilz


MODEL_FN = f"{mz.SRC_PATH}convemnist2.pth"

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
        self.epochs_per_live = 3

        self.model = modelz.ConvNet(out_dim=62)
        self.model.load_state_dict(torch.load(MODEL_FN))

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1e-3)

        self.live_infer_button = Button(
            self.root, text="toggle live infer", command=self.live_infer_toggle
        )
        self.live_infer_button.grid(row=0, column=0)

        self.infer_button = Button(self.root, text="infer", command=self.infer)
        self.infer_button.grid(row=0, column=1)

        self.info_button = Button(self.root, text="info", command=self.info)
        self.info_button.grid(row=0, column=2)

        self.erase_button = Button(self.root, text="erase all", command=self.clear)
        self.erase_button.grid(row=0, column=3)

        self.auto_erase_button = Button(
            self.root, text="auto_erase", command=self.toggle_auto_erase
        )
        self.auto_erase_button.grid(row=0, column=4)

        self.save_button = Button(self.root, text="save", command=self.save)
        self.save_button.grid(row=0, column=5)

        self.train_button = Button(
            self.root, text="live_train", command=self.live_train
        )
        self.train_button.grid(row=0, column=6)

        self.c = Canvas(
            self.root, bg=config["bg"], width=config["width"], height=config["height"]
        )
        self.c.grid(
            row=1, columnspan=7,
        )

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
        if self.live_infer:
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
        self.state_bounds.append(utilz.stroke_bounds(s))
        self.cur_stroke = []
        if self.live_infer:
            self.infer()
        if self.auto_erase:
            self.clear()

    def clear(self):
        self.chars = []
        self.state_bounds = []
        self.state = []
        self.img = None
        self.c.delete("all")

    def save(self, targets=None):

        self.ps = self.c.postscript(colormode="gray")
        self.img = save_canvas(self.ps, save=True)
        self.chars = utilz.get_chars(self.img, self.state_bounds)

        if targets:
            self.pil_chars = [
                utilz.save_char(char, f"{time.asctime()}_{fn}")
                for i, (char, fn) in enumerate(zip(self.chars, targets))
            ]
        else:
            self.pil_chars = [
                utilz.save_char(char, str(i)) for i, char in enumerate(self.chars)
            ]
        # self.infer()

    def info(self):
        print(f"state: {self.state}")
        print(f"chars: {self.chars}")
        print(f"state_bounds: {self.state_bounds}")
        print(f"num_strokes: {len(self.state)}")

        if self.chars:
            print(f"img shape: {self.img.shape}")

    def infer(self):
        if len(self.chars) == 0:
            self.save()
        pred_str = []
        for i, char in enumerate(self.pil_chars):
            pred = utilz.infer_char(self.model, char)
            pred_str.append(pred)

        print(f"pred_str: {pred_str}")

    def live_train(self):

        if len(self.chars) == 0:
            self.save()

        print("type in what you wrote")
        x = input()

        if len(x) != len(self.chars):
            print(f"len(x): {len(x)}, len(chars): {len(self.chars)}")
            print("multi stroke chars not supported yet")
            return

        npclasses = np.array(mz.EMNIST_CLASSES)
        target_idxs = []
        for elt in x:
            idx = np.where(npclasses == elt)[0][0]
            target_idxs.append(torch.tensor(idx).view(-1))

        results = {}
        all_targets = []
        all_preds = []
        all_losses = []

        data_chars = list(map(utilz.np_to_emnist_tensor, self.chars))

        for epoch in range(self.epochs_per_live):
            targets = []
            preds = []
            losses = []
            for char, target in zip(data_chars, target_idxs):
                target_char = mz.EMNIST_CLASSES[target]
                self.optimizer.zero_grad()
                output = self.model(char)
                pred_char = utilz.emnist_val(output)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                targets.append(target_char)
                preds.append(pred_char)

            all_targets += targets
            all_preds += preds
            all_losses += losses

            results[epoch] = list(zip(targets, preds, losses))

        print(all_targets)
        res = list(zip(all_targets, all_preds, all_losses))
        for elt in res:
            print(elt)

        torch.save(self.model.state_dict(), MODEL_FN)
        print(f"model saved to {MODEL_FN}")
        self.save(targets=targets)


def save_canvas(ps, fn="test", save=False):
    img = Image.open(io.BytesIO(ps.encode("utf-8")))

    if save:
        img.save(f"{mz.IMGS_PATH}{fn}.jpg", "jpeg")
    return np.asarray(img)


if __name__ == "__main__":
    x = Paint()
