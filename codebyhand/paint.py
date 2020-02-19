"""
todo single pen click throws index error

cursive wont work for the bounding box grabber 
https://stackoverflow.com/questions/41940945/saving-canvas-from-tkinter-to-file

"""

from tkinter import *
from tkinter.colorchooser import askcolor

import io
from PIL import Image, ImageTk
import numpy as np
import pyscreenshot as ImageGrab
import torch
import torchvision


config = {"width": 1400, "height": 500, "pen_radius": 5, "bg": "white"}

PATH = "/home/sippycups/programming/repos/usbtablet/assets/imgs/"


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

        self.pen_button = Button(self.root, text="pen", command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.snap_button = Button(self.root, text="snap", command=self._snapCanvas)
        self.snap_button.grid(row=0, column=1)

        self.info_button = Button(self.root, text="info", command=self.info)
        self.info_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text="erase all", command=self.clear)
        self.eraser_button.grid(row=0, column=3)

        self.save_button = Button(self.root, text="save", command=self.save)
        self.save_button.grid(row=0, column=4)

        self.c = Canvas(
            self.root, bg=config["bg"]
        )  # , width=config['width'], height=config['height'])
        self.c.grid(
            row=1, columnspan=5,
        )

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = config["pen_radius"]
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind("<B1-Motion>", self.paint)
        self.c.bind("<ButtonRelease-1>", self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        # self.activate_button(self.eraser_button, eraser_mode=True)
        self.c.delete("all")

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        paint_color = "white" if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=self.line_width,
                fill=paint_color,
                capstyle=ROUND,
                smooth=TRUE,
                splinesteps=36,
            )

        self.old_x = event.x
        self.old_y = event.y
        self.cur_stroke.append([self.old_x, self.old_y])
        # print(f'x: {self.old_x}, y: {self.old_y}')

    def reset(self, event):
        self.old_x, self.old_y = None, None
        s = np.array(self.cur_stroke)
        self.state.append(s)
        self.state_bounds.append(stroke_bounds(s))
        self.cur_stroke = []

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
        [save_char(char, str(i)) for i, char in enumerate(self.chars)]

        # print(f'img: {self.img}')

    def _snapCanvas(self):
        print("\n def _snapCanvas(self):")
        canvas = self.info()  # Get Window Coordinates of Canvas
        self.grabcanvas = ImageGrab.grab(bbox=canvas)
        self.grabcanvas.show()

    def _save(self):
        self.grabcanvas.save("out.jpg")

    def info(self):
        # broken
        print(f"state: {self.state}")
        print(f"chars: {self.chars}")

        print(f"state_bounds: {self.state_bounds}")
        print(f"num_strokes: {len(self.state)}")
        print(f"img shape: {self.img.shape}")
        print(f"self.cv.winfo_rootx() = {self.c.winfo_rootx()}")
        print(f"self.c.winfo_rooty() = {self.c.winfo_rooty()}")
        print(f"self.c.winfo_x() ={self.c.winfo_x()}")
        print(f"self.c.winfo_y() ={self.c.winfo_y()}")
        print(f"self.c.winfo_width() ={self.c.winfo_width()}")
        print(f"self.c.winfo_height() ={self.c.winfo_height()}")
        x = self.c.winfo_rootx() + self.c.winfo_x()
        y = self.c.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        box = (x, y, x1, y1)
        print("box = ", box)
        return box


def norm_stroke(s: np.ndarray) -> np.ndarray:
    return s - s[0]


def stroke_bounds(s: np.ndarray) -> np.ndarray:
    xmin = s[:, 0].min()  # - 200
    xmax = s[:, 0].max()  # - 200

    ymin = s[:, 1].min()  # - 150
    ymax = s[:, 1].max()  # - 150

    return (xmin, xmax), (ymin, ymax)


def adj(n: int, m: int):
    return int((n - m) / 2)


def fix_bound(b, img):
    shape = img.shape
    h_adj = adj(config["height"], shape[0])
    w_adj = adj(config["width"], shape[1])


def save_canvas(ps, fn="test", save=False):
    img = Image.open(io.BytesIO(ps.encode("utf-8")))
    if save:
        img.save(
            f"{PATH}{fn}.jpg", "jpeg"
        )  # , height=config['height'], width=config['width'])

    return np.asarray(img)


def adj2(x: int) -> int:
    return int(0.75 * x)


def char_from_img(img, b, pad: int = 5):
    x0, x1 = b[0]
    y0, y1 = b[1]

    x0, x1, y0, y1 = [adj2(x) for x in [x0, x1, y0, y1]]

    print(img.shape)
    print(f"xs {x0, x1}")
    print(f"ys {y0, y1}")
    return img[y0 - pad : y1 + pad, x0 - pad : x1 + pad]
    # return img[xs[0]:xs[1], ys[0]:ys[1]]


def get_chars(img, bounds):
    return [char_from_img(img, b) for b in bounds]


def save_char(char: np.array, fn: str):
    shape = char.shape
    print(f"char shape: {char.shape}")
    im = Image.fromarray(char)
    im.save(f"{PATH}{fn}.png")


if __name__ == "__main__":
    x = Paint()
