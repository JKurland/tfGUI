from layer_base import *

def use_variable(scope_name, var_name, shape):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var_name, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var_name)

        if not compare_shapes(v.get_shape().as_list(), shape):
            raise Exception('Shared variables have different shapes ' +
                            scope_name + '/' + var_name)

        return v
        

class Linear(LayerBase):
    """
    class for linear projections
    """

    def __init__(self, name, canvas, size):
        LayerBase.__init__(self, name, canvas, [None,None], allow_multiple_inputs =
                           True)
        self._size = size

    
    def proc(self, inputs):
        in_size = inputs.get_shape().as_list()[-1]

        w = use_variable(self.name, 'weights', [in_size, self._size])
        b = use_variable(self.name, 'biases', [self._size])

        self._variables.append(w)
        self._variables.append(b)
        with tf.variable_scope(self.name):
            return tf.matmul(inputs, w) + b


class Relu(LayerBase):
    """
    class for rectified linear units
    """

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas, None, allow_multiple_inputs = True)


    def proc(self, inputs):

        with tf.variable_scope(self.name):
            return tf.nn.relu(inputs, 'relu')

class Constant(LayerBase):
    """layer which outputs a constant numpy array"""

    def __init__(self, name, canvas, value):
        LayerBase.__init__(self, name, canvas, [-1]) #required shape of [-1] means no
        #input can be accepted
        self.real = True
        self.output = tf.convert_to_tensor(value, tf.float32)
    
    def reset_real(self):
        self.real = True

#because self.real is set to true proc and pre_proc will never be called




    



