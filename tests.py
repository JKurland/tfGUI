import tensorflow as tf
import numpy as np
import tkinter as tk
from tests import *
from simple_layers import Linear, Constant, Relu, DragManager

def simple_network_test(canvas)
    c = Constant('input', canvas, np.random.rand(1,3))
    c2 = Constant('input2', canvas, np.random.rand(1,4))