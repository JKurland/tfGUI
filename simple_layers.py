from layer_base import *
import numpy as np
from matplotlib import pyplot as plt


def use_variable(scope_name, var_name, shape, init = None):
    with tf.variable_scope(scope_name) as scope: 
        try:
            if init is not None:
                if isinstance(init, tf.Tensor):
                    v = tf.get_variable(var_name, initializer = init)
                else:
                    v = tf.get_variable(var_name, shape, initializer = init)
            else:
                v = tf.get_variable(var_name, shape)
        except ValueError as inst:
            scope.reuse_variables()
            if init is not None:
                v = tf.get_variable(var_name, initializer = init)
            else:
                v = tf.get_variable(var_name)
        
        
        if not compare_shapes(v.get_shape().as_list(), shape):
            raise Exception('Shared variables have different shapes ' +
                            scope_name + '/' + var_name +
                           '. Got shapes {}, {}'.format(shape,
                                                        v.get_shape().as_list()))

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
            return tf.matmul (inputs, w) + b


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

        self._variables['value'] = value
   
    def proc(self, inputs):
        val = self._variables['value']
        return tf.convert_to_tensor(eval(val), tf.float32)

class Print(LayerBase):
    """layer which prints its input when run"""

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas)
        self.t = 0
        self._variables['Frequency'] = 1

    def run(self,session):
        if self.t%int(self._variables['Frequency']) == 0:
            print(session.run(self.output))
        self.t += 1

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
 

class Plot(LayerBase):

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas,
                           input_share = [False],
                           input_shapes = [[]])
        
        self.log = []
        self.x = []
        self._variables['Frequency'] = 1
        self._variables['Title'] = self.name
        self._variables['y label'] = ''
        self.t = 0
        
        #name is chosen initially so that the first Plot object is called Plot
        #the second is Plot0, the third Plot1 etc... take advantage of that
        #to number the figures, very hacky, should be changed
        if name == 'Plot':
            num = 1
        else:
            num = int(name[4:])+2

        self.fig_num = num 


    def run(self, session):
        if self.t%int(self._variables['Frequency']) == 0:
            plt.figure(self.fig_num)
            plt.clf()

            plt.title(self._variables['Title'])
            plt.xlabel('Step')
            plt.ylabel(self._variables['y label'])
            self.x.append(self.t)
            self.log.append(session.run(self.output))
            plt.plot(self.x, self.log)
            plt.draw()
            plt.pause(0.01)
        
        self.t +=1

    def reset_real(self):
        LayerBase.reset_real(self)
        plt.figure(self.fig_num)
        plt.clf()
        self.t = 0
        self.log = []
        self.x = []


class End(LayerBase):
     #class for running everything beneath it

    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas)
        self._variables['iters'] = 1

    @property
    def repeat(self):
        return int(self._variables['iters'])

    @repeat.setter
    def repeat(self, value):
        self._variables['iters'] = value

    def check_compatible(self, inputs):
        return

    def pre_proc(self, inputs):
        return None

    def proc(self, inputs):
        return None


class Sample(LayerBase):
    def __init__(self, name, canvas):
        LayerBase.__init__(self, name, canvas)
        self._variables['N'] = '64'

    def proc(self, inputs):
        i = inputs['main']
        shape = i.get_shape().as_list()
        dims = len(shape)
        total_size = shape[0]
        N = eval(self._variables['N'])

        shuffle = tf.random_shuffle(i)
        t = use_variable(self.name, 'step', [], init = tf.constant(0))

        if total_size<N:
            N = total_size
            inc_t = tf.assign(t, 0)
        else:
            inc_t = tf.assign(t, tf.mod(t+N, total_size-N))
        
        with tf.control_dependencies([inc_t]):
            begin = [t] + (dims-1)*[0]
            size = [N] + (dims-1)*[-1]
            output = tf.slice(shuffle, begin, size)
        
        return output


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



# if you add a new layer class, add a shortcut to it here, in __init__
class DragManager():
    
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
                              't':Trainer,
                              'o':Plot,
                              'e':End,
                              's':Sample}
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
            
            self.move_by_id(ID[0], event.x - self.mouse_x,
                            event.y-self.mouse_y)
        
            self.mouse_x = event.x
            self.mouse_y = event.y


    def move_by_id(self, ID, d_x, d_y):
        widgets = self.canvas.find_withtag(ID)
        for w in widgets:
            self.canvas.move(w, d_x, d_y)        
        arrows_to = self.canvas.find_withtag('e' + ID)
        for a in arrows_to:
            X = self.canvas.coords(a)
            x0 = X[0]
            y0 = X[1]
            
            x1 = X[2]
            y1 = X[3]
            
            x1 += d_x 
            y1 += d_y 

            self.canvas.coords(a, x0, y0, x1, y1)
        
        arrows_to = self.canvas.find_withtag('s' + ID)
        for a in arrows_to:
            X = self.canvas.coords(a)
            x0 = X[0]
            y0 = X[1]

            x1 = X[2]
            y1 = X[3]

            x0 += d_x 
            y0 += d_y 

            self.canvas.coords(a, x0, y0, x1, y1)

    def move_by_obj(self, obj, d_x, d_y):
        ID = self.obj2id[obj]
        self.move_by_id(ID, d_x, d_y)
    
    def place_by_obj(self, obj, x, y):
        #place an object at an absolute location, anchored at the top left
        X = obj.coords
        
        x_origin = X[0]
        y_origin = X[1]

        self.move_by_obj(obj, x - x_origin, y - y_origin)

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
    
