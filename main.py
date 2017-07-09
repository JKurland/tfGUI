import tensorflow as tf
import numpy as np
import tkinter as tk
from tests import *
from simple_layers import Linear, Constant, Relu, DragManager
from manager import *

root = tk.Tk()

canvas = tk.Canvas(root, width = 400, height = 400)

manager = Manager(canvas)

#simple_network_test(canvas, drag_manager)

canvas.pack(fill = tk.BOTH, expand = tk.YES)


tk.mainloop()
