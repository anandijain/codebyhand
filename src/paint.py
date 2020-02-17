"""
todo single pen click throws index error

"""

from tkinter import *
from tkinter.colorchooser import askcolor

import io
from PIL import Image, ImageTk
import numpy as np

config = {
    "width" : 1200,
    "height" : 400,
    "pen_radius": 5,
    'bg': 'white'
}

PATH = '/home/sippycups/programming/repos/usbtablet/assets/imgs/'

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.state = []
        self.state_dict = {}
        self.cur_stroke = []
        self.img = None
        self.state_bounds = []

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.info_button = Button(self.root, text='info', command=self.info)
        self.info_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='erase all', command=self.clear)
        self.eraser_button.grid(row=0, column=3)

        self.save_button = Button(self.root, text='save', command=self.save)
        self.save_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg=config['bg'], width=config['width'], height=config['height'])
        self.c.grid(row=1, columnspan=5)
        
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = config['pen_radius']
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)


    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

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
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)

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
        print(f'state: {self.state}')
        print(f'state_bounds: {self.state_bounds}')
        self.state_bounds = []
        self.state = []
        self.c.delete('all')

    def save(self):
        self.img = save_canvas(self.c, save=True)
        print(f'img: {self.img}')

    def info(self):
        # broken
        print(f'state: {self.state}')
        print(f'state_bounds: {self.state_bounds}')
        print(f'num_strokes: {len(self.state)}')


def norm_stroke(s: np.ndarray) -> np.ndarray:
    return s - s[0]


def stroke_bounds(s: np.ndarray) -> np.ndarray:
    xmin = s[:, 0].min()
    ymin = s[:, 1].min()
    xmax = s[:, 0].max()
    ymax = s[:, 1].max()
    return (xmin, xmax), (ymin, ymax)


def save_canvas(c:Canvas, fn='test', save=False):
    ps = c.postscript(colormode='gray')
    img = Image.open(io.BytesIO(ps.encode('utf-8')))
    if save:
        img.save(f'{PATH}{fn}.jpg', 'jpeg')
    return img

def char_from_img(img, b):
    xs = b[0]
    ys = b[1]
    return img[xs[0]:xs[1], ys[0]:ys[1]]


if __name__ == '__main__':
    Paint()
