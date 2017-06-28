import tensorflow as tf
import numpy as np

from simple_layers import Linear, Constant, Relu

c = Constant('input', np.random.rand(10,20))
c2 = Constant('input2', np.random.rand(10,33))

lin = Linear('linear', 23)

relu = Relu('relu')


relu.add_input(lin)
lin.add_input(c)
lin.add_input(c2)

relu.make_real()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
a = sess.run(relu.output)

print(a.shape)


