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

    def __init__(self, name, canvas, size=20):
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

    def __init__(self, name, canvas, value=0):
        LayerBase.__init__(self, name, canvas, [-1]) #required shape of [-1] means no
        #input can be accepted
        self.real = True
        self.output = tf.convert_to_tensor(value, tf.float32)
    
    def reset_real(self):
        self.real = True

#because self.real is set to true proc and pre_proc will never be called



class DragManager():
    """
    class for an object which managesthe dragging of objects
    """

    def __init__(self, canvas):
        self.widget = None
        self.canvas = canvas
        self.mouse_x = None
        self.mouse_y = None
        self.id2obj = {}
        self.obj2id = {}
        self.names = []
        self.shortcut_dict = {'l':Linear,
                              'c':Constant,
                              'r':Relu}


    def on_click(self, event):
        self.widget = self.canvas.find_closest(event.x, event.y, halo = 5)[0]
        self.mouse_x = event.x
        self.mouse_y = event.y

    def on_move(self, event):
        if self.widget:
            ID = self.canvas.gettags(self.widget)
            if len(ID) >1 : #an arrow got selected
                return

            widgets = self.canvas.find_withtag(ID)
            for w in widgets:
                self.canvas.move(w, event.x - self.mouse_x,
                                 event.y - self.mouse_y)
            
            arrows_to = self.canvas.find_withtag('e' + ID[0])
            for a in arrows_to:
                X = self.canvas.coords(a)
                x0 = X[0]
                y0 = X[1]
                
                x1 = X[2]
                y1 = X[3]
                
                x1 += event.x - self.mouse_x
                y1 += event.y - self.mouse_y

                self.canvas.coords(a, x0, y0, x1, y1)
            
            arrows_to = self.canvas.find_withtag('s' + ID[0])
            for a in arrows_to:
                X = self.canvas.coords(a)
                x0 = X[0]
                y0 = X[1]

                x1 = X[2]
                y1 = X[3]

                x0 += event.x - self.mouse_x
                y0 += event.y - self.mouse_y

                self.canvas.coords(a, x0, y0, x1, y1)
 
            
            self.mouse_x = event.x
            self.mouse_y = event.y
    
    def on_release(self, event):
        self.widget = None
        self.mouse_x = None
        self.mouse_y = None

    def on_r_click(self, event):
        self.widget = self.canvas.find_closest(event.x, event.y, halo = 5)[0]
        self.widget = self.canvas.gettags(self.widget)[0] 

    def on_r_release(self, event):
        other_widget = self.canvas.find_closest(event.x, event.y, halo = 5)[0]
        other_widget = self.id2obj[self.canvas.gettags(other_widget)[0]]
        other_widget.add_input(self.id2obj[self.widget])

    def on_key(self, event):
        print(event.char)
        try:
            Class = self.shortcut_dict[event.char]
        except:
            return
        name = Class.__name__
        i = 0
        new_name = '%s%i'%(name, i)
        if name in list(self.names):
            while new_name in list(self.names):
                i+=1
                new_name = '%s%i'%(name, i)

            name = new_name
        
        
        new_layer = Class(name, self.canvas)
        self.id2obj[new_layer.id] = new_layer
        self.obj2id[new_layer] = new_layer.id
        self.names.append(new_layer.name)
    

        



