from tkinter import *
from tkinter.colorchooser import askcolor

config = {
    "width" : 1200,
    "height" : 400,
    "pen_radius": 5
}


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='erase all', command=self.clear)
        self.eraser_button.grid(row=0, column=3)

        self.c = Canvas(self.root, bg='white', width=config['width'], height=config['height'])
        self.c.grid(row=1, columnspan=4)

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
        self.stroke = [] # list of x,y tuples 

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
        self.stroke.append((self.old_x, self.old_y))
        # print(f'x: {self.old_x}, y: {self.old_y}')

    def reset(self, event):
        self.old_x, self.old_y = None, None
        print(f'self.stroke : {self.stroke}')
        self.stroke = []  # list of x,y tuples


    def clear(self):
        self.c.delete('all')



if __name__ == '__main__':
    Paint()
