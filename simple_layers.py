from layer_base import *
import numpy as np



def use_variable(scope_name, var_name, shape):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var_name, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var_name)
        
        
        if not compare_shapes(v.get_shape().as_list(), shape):
            print(shape)
            print(v.get_shape().as_list())
            raise Exception('Shared variables have different shapes ' +
                            scope_name + '/' + var_name)

        return v
        

class Linear(LayerBase):
    """
    class for linear projections
    """

    def __init__(self, name, canvas, size=20):
        LayerBase.__init__(self, name, canvas, input_shapes = [[None, -1]])
        #None forces all inputs to be the same size, -1 allows inputs to be
        #different
        self._variables['size'] = size
    
    def proc(self, inputs):
        self._size = int(self._variables['size'])
        inputs = inputs['main']
        in_size = inputs.get_shape().as_list()[-1]

        w = use_variable(self.name, 'weights', [in_size, self._size])
        b = use_variable(self.name, 'biases', [self._size])

        self._tf_vars.append(w)
        self._tf_vars.append(b)
        with tf.variable_scope(self.name):
            return tf.matmul(inputs, w) + b


class Relu(LayerBase):
    """
    class for rectified linear units
    """

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas)


    def proc(self, inputs):
        inputs = inputs['main']
        with tf.variable_scope(self.name):
            return tf.nn.relu(inputs, 'relu')


class Constant(LayerBase):
    """layer which outputs a constant numpy array"""
    

    def __init__(self, name, canvas, value='np.random.rand(5,10)'):
        LayerBase.__init__(self, name, canvas, input_number = 0)
        self.value = value
   
    def proc(self, inputs):
        return tf.convert_to_tensor(eval(self.value), tf.float32)

class Print(LayerBase):
    """layer which prints its input when run"""

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas)

    def run(self,session):
        print(session.run(self.output))


class MSELoss(LayerBase):

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name,canvas, input_number = 2,
                           input_names = ['estimates','targets'],
                           input_share = [True, True],
                           input_shapes = [None, None],
                           input_require = [True, True])

    def proc(self, inputs):
        targets = inputs['targets']
        est = inputs['estimates']
        output = tf.reduce_mean(tf.square(est-targets))
        return output

class FileReader(LayerBase):

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas, input_number = 0)

        self._variables['path'] = 'data'

    def proc(self, inputs):
        path = self._variables['path']
        data = np.genfromtxt(path, delimiter = ',')
        return tf.convert_to_tensor(data,tf.float32)

class Trainer(LayerBase):

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas, input_names = ['Loss'])
        self._variables['iters'] == 10
        self._train_op = None 
    

    @property
    def repeat(self):
        return int(self._variables['iters'])
    
    @repeat.setter
    def repeat(self, value):
        self._variables['iters'] = value

    def proc(self, inputs):
        loss = inputs['Loss']
        self._train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return loss

    def run(self, session):
        session.run(self._train_op)


def add_layer(Class, man):
    name = Class.__name__
    i = 0
    new_name = '%s%i'%(name, i)
    if name in list(man.names):
        while new_name in list(man.names):
            i+=1
            new_name = '%s%i'%(name, i)

        name = new_name
        
        
    new_layer = Class(name, man.canvas)
    man.id2obj[new_layer.id] = new_layer
    man.obj2id[new_layer] = new_layer.id
    man.names.append(new_layer.name)
    man.layers.append(new_layer)




class DragManager():
    """
    class for an object which managesthe dragging of objects
    """

    def __init__(self, canvas):
        self.widget = None 
        self.sess = tf.Session()
        self.canvas = canvas
        self.mouse_x = None
        self.mouse_y = None
        self.id2obj = {}
        self.obj2id = {}
        self.names = []
        self.layers = []
        self.logs = {}
        self.shortcut_dict = {'l':Linear,
                              'c':Constant,
                              'r':Relu,
                              'p':Print,
                              'm':MSELoss,
                              'f':FileReader,
                              't':Trainer}
        self.tags = None 
    
    def find_close(self, x, y, range = 1):
        
        close = self.canvas.find_overlapping(x-range, y-range, x+range, y+range)
        if len(close) == 0:
            return [None]
        else:
            return close


    def on_click(self, event):
        self.widget = self.canvas.find_closest(event.x, event.y,halo = 5)[0]
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        self.x = event.x
        self.y = event.y

    def on_move(self, event):
        if self.widget:
            ID = self.canvas.gettags(self.widget)
            if 'arrow' in ID: #an arrow got selected
                return

            widgets = self.canvas.find_withtag(ID[0])
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
        
        widget = self.find_close(event.x, event.y)[0]
        
        if widget == None or self.widget == None:
            for l in self.layers:
                if l.showing_v_win:
                    l.hide_variable_window()
        else:
            first_tags = self.canvas.gettags(self.widget)
            tags = self.canvas.gettags(widget) 
            
            dist = (self.x - event.x)**2 + (self.y - event.y)**2

            if first_tags[0] == tags[0] and dist < 25: #we clicked on one object
                            
                if 'arrow' in tags:
                    return

                obj = self.id2obj[tags[0]]
                if not obj.showing_v_win:
                    obj.show_variable_window()
                else:
                    obj.hide_variable_window()
    

        self.widget =  None
        self.mouse_x = None
        self.mouse_y = None



    def on_double(self, event):
        #self.widget is set by the on_click event which is called first
        widget = self.canvas.find_closest(event.x, event.y, halo = 5)[0]

        tags = self.canvas.gettags(widget)
        
        if 'arrow' not in tags:
            obj = self.id2obj[tags[0]]
        else:
            return
        
        if not obj.real:
            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()
            #as the graph is being cleared we can't keep anything from
            #previous realifying so reset. Reseting the obj will reset
            #everything in the same connected region as the obj
            obj.reset_real()
            if obj.make_real() == 0:
                print("realifying failed")
                return 
            init = tf.global_variables_initializer()
            self.sess.run(init)
        for _ in range(obj.repeat):
            for l in self.layers:
                if l.real:
                    l.run(self.sess)


    def on_r_click(self, event):
        self.widget = self.canvas.find_closest(event.x, event.y, halo = 5)[0] 
        self.tags = self.canvas.gettags(self.widget)

    def on_r_release(self, event):
        other_widget = self.canvas.find_closest(event.x, event.y, halo = 5)[0]
        other_tags = self.canvas.gettags(other_widget) 
        if 'arrow' in other_tags or 'arrow' in self.tags:
            return #if either object is an arrow return
        
        if 'input' not in other_tags:
            return #the target is not an input so return

        obj = self.id2obj[self.canvas.gettags(self.widget)[0]]
        other_obj = self.id2obj[self.canvas.gettags(other_widget)[0]]
        
        socket = other_tags[2]
        
        for k in obj._inputs.keys():
            if other_obj in obj._inputs[k]:
                obj.remove_input(other_obj,k)
                return 

        if obj in other_obj._inputs[socket]:
            other_obj.remove_input(obj,socket)

        else:
            other_obj.add_input(obj,socket)
        
    def on_key(self, event):
        print(event.char)
        
        for l in self.layers:
            if l.showing_v_win:
                return

        try:
            Class = self.shortcut_dict[event.char]
        except:
            return
        add_layer(Class, self)
      



