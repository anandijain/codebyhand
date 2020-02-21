"""
todo single pen click throws index error

cursive wont work for the bounding box grabber 
https://stackoverflow.com/questions/41940945/saving-canvas-from-tkinter-to-file

"""
import io
import time
from tkinter import *
from tkinter.colorchooser import askcolor

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image, ImageTk
from torch.utils.data import DataLoader, Dataset

from codebyhand import loaderz
from codebyhand import macroz as mz
from codebyhand import modelz, train, utilz

MODEL_FN = f"{mz.SRC_PATH}spatial_transformer_net.pth"
VISUALIZE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

config = {
    "width": 1400, 
    "height": 500,
    "pen_radius": 8, 
    "bg": "white", 
    "pen": "black",
    "epochs" : 3
}
plt.ion()


class Paint(object):

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

        self.model = modelz.SpatialTransformerNet(out_dim=62).to(device)
        self.model.load_state_dict(torch.load(MODEL_FN))

        self.optimizer = optim.Adadelta(self.model.parameters())

        self.live_infer_button = Button(
            self.root, text="toggle live infer", command=self.live_infer_toggle
        )
        self.live_infer_button.grid(row=0, column=0)

        self.infer_button = Button(self.root, text="infer", command=self.infer)
        self.infer_button.grid(row=0, column=1)

        self.info_button = Button(self.root, text="info", command=self.info)
        self.info_button.grid(row=0, column=2)

        self.erase_button = Button(
            self.root, text="erase all", command=self.clear)
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
        self.color = config['pen']
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
        if self.auto_erase:
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

        if targets is not None:
            self.pil_chars = [
                utilz.save_labeled_char(char, t, f"{time.asctime()}_{t}")
                for char, t in zip(self.chars, targets)
            ]

        else:
            self.pil_chars = [
                utilz.save_char(char, str(i)) for i, char in enumerate(self.chars)
            ]

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
            pred = utilz.infer_char(self.model, char, device)
            pred_str.append(pred)

        print(f"pred_str: {pred_str}")

    def live_train(self):
        if len(self.chars) == 0:
            self.save()

        target_idxs, target_chars = get_user_labels(self.chars)
        dataset, loader = chars_to_data(self.chars, target_idxs)
        self.d = {
            "data": dataset,
            "loader": loader,
            "model": self.model,
            "optimizer": self.optimizer,
        }
        all_preds, all_losses = train.train(
            self.d, config['epochs'], True, MODEL_FN
        )
        results = list(
            zip(target_chars * config['epochs'], all_preds, all_losses))

        print("results:")
        for elt in results:
            print(elt)

        if VISUALIZE:
            visualize_stn(self.d)
            plt.ioff()
            plt.show()

        torch.save(self.model.state_dict(), MODEL_FN)
        print(f"model saved to {MODEL_FN}")
        self.save(targets=target_chars)


def save_canvas(ps, fn="test", save=False):
    img = Image.open(io.BytesIO(ps.encode("utf-8")))
    if save:
        img.save(f"{mz.IMGS_PATH}{time.asctime()}{fn}.png", "png")
    return np.asarray(img)


def target_info(s: str):
    classes = np.array(mz.EMNIST_CLASSES)
    target_idxs = []
    target_chars = []
    for elt in s:
        idx = np.where(classes == elt)[0][0]
        target_idxs.append(torch.tensor(idx))  # .view(-1))
        target_chars.append(mz.EMNIST_CLASSES[idx])
    return target_idxs, target_chars


# def to_points(state:list):
def chars_to_data(chars, target_idxs):

    data_chars = list(map(utilz.np_to_emnist_tensor, chars))
    data = list(zip(data_chars, target_idxs))
    dataset = loaderz.Chars(data)
    # x, y = dataset[0][0], dataset[0][1]

    # print(f'example: {x}')
    # print(f'example: {y}')
    loader = DataLoader(dataset)
    return dataset, loader


def get_user_labels(chars):
    print("type in what you wrote")
    x = input()
    if len(x) != len(chars):
        print(f"len(x): {len(x)}, len(chars): {len(chars)}")
        print("multi stroke chars not supported yet")
        return

    target_idxs, target_chars = target_info(x)
    return target_idxs, target_chars


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(d):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(d['loader']))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = d['model'].stn(data).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor)
        )

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        axarr[1].imshow(out_grid)
        axarr[1].set_title("Transformed Images")


if __name__ == "__main__":
    x = Paint()
