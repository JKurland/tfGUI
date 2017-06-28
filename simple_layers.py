from layer_base import *

def use_variable(scope_name, var_name, shape):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var_name, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var_name)

        if not compare_shapes(v.get_shape().as_list(), shape):
            raise Exception('Shared variables have different shapes' +
                            scope_name + '/' + var_name)

        return v
        

class LinLayer(LayerBase):
    """
    class for linear projections
    """

    def __init__(self, size, name):
        LayerBase.__init__(self, name, [None,None], allow_multiple_inputs =
                           True)
        self._size = size

    
    def proc(self, inputs):
        in_size = inputs.get_shape().as_list()[-1]

        w = use_variable(self.name, 'weights', [in_size, self.size])
        b = use_variable(self.name, 'biases', [self.size])

        self._variables.append(w)
        self._variables.append(b)
        with tf.variable_scope(self.name):
            return tf.matmul(inputs, w) + b



