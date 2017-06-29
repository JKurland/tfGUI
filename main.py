import tensorflow as tf
import numpy as np
import tkinter as tk

from simple_layers import Linear, Constant, Relu, DragManager

root = tk.Tk()

canvas = tk.Canvas(root, width = 400, height = 400)

drag_manager = DragManager(canvas)

canvas.bind("<Button-1>", drag_manager.on_click)
canvas.bind("<B1-Motion>", drag_manager.on_move)
canvas.bind("<ButtonRelease-1>", drag_manager.on_release)   
canvas.bind_all("<Button-3>", drag_manager.on_r_click)
canvas.bind_all("<ButtonRelease-3>", drag_manager.on_r_release)
canvas.bind_all("<KeyPress>", drag_manager.on_key)


#c = Constant('input',canvas, np.random.rand(1,3))
#c2 = Constant('input2',canvas, np.random.rand(1,4))

#lin = Linear('linear',canvas, 7)
#lin2 = Linear('linear2',canvas, 7)

#relu = Relu('relu', canvas)
#relu2 = Relu('relu2', canvas)

#relu.add_input(lin)
#relu2.add_input(lin2)

#lin.add_input(c)
#lin.add_input(c)
#lin.add_input(c2)

#lin2.add_input(relu)
#lin2.add_input(lin)

#relu2.make_real()

canvas.pack(fill = tk.BOTH, expand = tk.YES)


#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#a = sess.run(relu2.output)

#print(a)

tk.mainloop()
