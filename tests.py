import tensorflow as tf
import numpy as np
import tkinter as tk
from simple_layers import Linear, Constant, Relu, DragManager, add_layer

class Dummy_event():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

def simple_network_test(canvas, man):

    add_layer(Linear, man)
    origin_click = Dummy_event()
    man.on_click(origin_click)
    move_click = Dummy_event(x=100, y=100)
    man.on_move(move_click)


    # add_layer(Linear, man)


