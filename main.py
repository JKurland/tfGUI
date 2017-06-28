import tensorflow as tf
import numpy as np

from simple_layers import Linear, Constant, Relu

c = Constant('input', np.random.rand(1,3))
c2 = Constant('input2', np.random.rand(1,4))

lin = Linear('linear', 7)
lin2 = Linear('linear', 7)

relu = Relu('relu')
relu2 = Relu('relu2')

relu.add_input(lin)
relu2.add_input(lin2)

lin.add_input(c)
lin.add_input(c2)

lin2.add_input(relu)

relu2.make_real()


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
a = sess.run(relu.output)

print(a)


