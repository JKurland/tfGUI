from layer_base import *
import numpy as np
from matplotlib import pyplot as plt
from numpy import median
    

class Linear(LayerBase):
    """
    class for linear projections
    """

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man, input_shapes = [[None, -1]])
        #None forces all inputs to be the same size, -1 allows inputs to be
        #different
        self._variables['size'] = 20
    
    def proc(self, inputs):
        self._size = int(self._variables['size'])
        inputs = inputs['main']
        in_size = inputs.get_shape().as_list()[-1]

        w = self.use_variable('weights', [in_size, self._size])
        b = self.use_variable('biases', [self._size])

        tf.add_to_collection('weights', w)
        tf.add_to_collection('biases', w)

        return tf.matmul (inputs, w) + b

class Relu(LayerBase):
    """
    class for rectified linear units
    """

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man)


    def proc(self, inputs):
        inputs = inputs['main']
        with tf.variable_scope(self.name):
            return tf.nn.relu(inputs, 'relu')

class Constant(LayerBase):
    """layer which outputs a constant numpy array"""
    

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man, input_number = 0)
        self.value = 'np.random.rand(5,10)'

        self._variables['value'] = self.value
   
    def proc(self, inputs):
        val = self._variables['value']
        return tf.convert_to_tensor(eval(val), tf.float32)

class Print(LayerBase):
    """layer which prints its input when run"""

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man)
        self.t = 0
        self.node = None 
        self._variables['Frequency'] = 1

    def proc(self, inputs):
        out = LayerBase.proc(self, inputs)
        self._request_node(out)
        self.node = out
        return out

    def run(self):

        if self.t%int(self._variables['Frequency']) == 0:
            print(self._get_node(self.node))
        self.t += 1

class MSELoss(LayerBase):

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name,canvas, man, input_number = 2,
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

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man, input_number = 0)

        self._variables['path'] = 'data'

    def proc(self, inputs):
        path = self._variables['path']
        data = np.genfromtxt(path, delimiter = ',')
        return tf.convert_to_tensor(data,tf.float32)

class Trainer(LayerBase):

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man,
                           input_names = ['Loss'],
                           input_shapes = [None])
        self._variables['iters'] = '10'
        self._variables['alpha'] = '1e-4'
        self._train_op = None 
        self.t = 0

    def pre_proc(self, inputs):
        i = inputs['Loss']
        output = {'Loss': tf.add_n(i)}

        return output

    def proc(self, inputs):
        loss = inputs['Loss']
        t = self.t
        al = eval(self._variables['alpha'])
        self._train_op = tf.train.AdamOptimizer(al).minimize(loss,
                                                    aggregation_method = 2)
        with tf.control_dependencies([self._train_op]):
            out = tf.identity(loss)
        self._request_node(out)
        self.node = out
        return loss

    def run(self): 
        self._get_node(self.node)
        if self.t < eval(self._variables['iters']):
            self.cont = 1
        else:
            self.cont = 0
            self.t = 0
        self.t+=1

    def reset(self):
        LayerBase.reset(self)
        self.t = 0

class Plot(LayerBase):

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man,
                           input_share = [False],
                           input_shapes = [[]])
        
        self.log = []
        self.x = []
        self._variables['Frequency'] = 1
        self._variables['Title'] = self.name
        self._variables['y label'] = 'Value'
        self.t = 0
        man.figs += 1
        #name is chosen initially so that the first Plot object is called Plot
        #the second is Plot0, the third Plot1 etc... take advantage of that
        #to number the figures, very hacky, should be changed
        self.fig_num = man.figs

    def proc(self, inputs):
        o = LayerBase.proc(self, inputs)
        self.node = inputs['main']

        self._request_node(self.node)
        return o

    def run(self):
        if self.t%int(self._variables['Frequency']) == 0:
            plt.figure(self.fig_num)
            plt.clf()

            plt.title(self._variables['Title'])
            plt.xlabel('Step')
            plt.ylabel(self._variables['y label'])
            self.x.append(self.t)
            self.log.append(self._get_node(self.node))
            plt.plot(self.x, self.log)
            
            y_max = 3*median(self.log)
            y_min = 0
            axes = plt.gca()
            axes.set_ylim([y_min, y_max])
            plt.draw()
            plt.pause(0.01)
        
        self.t +=1

    def reset(self):
        LayerBase.reset(self)
        plt.figure(self.fig_num)
        plt.clf()
        self.t = 0
        self.log = []
        self.x = []

class End(LayerBase):
     #class for running everything beneath it

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man)
        self._variables['iters'] = '1' 
        self.t = 0

    def check_inputs(self, inputs):
        return

    def pre_proc(self, inputs):
        return None

    def proc(self, inputs):
        return None 

    def run(self):
        if self.t < eval(self._variables['iters']):
            self.cont = 1
        else:
            self.cont = 0
            self.t = 0
        self.t+=1

    def reset(self):
        LayerBase.reset(self)
        self.t = 0

class Sample(LayerBase):
    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man)
        self._variables['N'] = '64'

    def proc(self, inputs):
        i = inputs['main']
        shape = tf.shape(i)
        dims = shape.get_shape().as_list()[0]
        total_size = shape[0]
        N = eval(self._variables['N'])

        #shuffle = tf.random_shuffle(i)
        t = self.use_variable('step', [], init = tf.constant(0))
        epoch = self.use_variable('epoch', [],
                                  init = tf.constant(0.0, dtype = tf.float32))

        n = tf.ceil(total_size/N)
        inc_epoch = tf.assign_add(epoch,tf.cast(1/n, tf.float32))
    

        new_t = tf.mod(t+N, total_size-N)
        inc_t = tf.cond(total_size<N, lambda: tf.assign(t,0),
                        lambda: tf.assign(t, new_t))
        
        N = tf.minimum(N, total_size)

        with tf.control_dependencies([inc_t, inc_epoch]):
            begin = [t] + (dims-1)*[0]
            size = [N] + (dims-1)*[-1]
            output = tf.slice(i, begin, size)
                 
        return output

class ExtractParam(LayerBase):
    
    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man, input_share = [False])
        self._variables['Param Name'] = ''
        
        self.show_variable_window()
        self.hide_variable_window()

    def proc(self, inputs):
        i = self._inputs['main'][0][0] #the input object
        if self._variables['Param Name'] == '':
            return tf.constant(0)
        else:
            n = self._variables['Param Name']
            
            for var in i._tf_vars:
                if var.name.split('/')[-1][:-2].lower() == n.lower():
                    output = var 
                    return tf.identity(output) 
            return tf.constant(0)

class ShowImage(LayerBase):
    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man, input_share = [False])
        man.figs += 1
        self.fig_num = man.figs
        self._variables['Title'] = self.name
        self._variables['vmin'] = ''
        self._variables['vmax'] = ''
        self._variables['Frequency'] = '1'
        self.t = 0

    def check_input(self, inputs):
        LayerBase.check_inputs(self, inputs)
        
        shape = input['main'][0].get_shape().as_list()
        dims = len(shape)
        if dims == 3:
            if shape[2] != 1 and shape[2]!=3:
                raise Exception('If input has 3 dimensions the third ' +
                                'dimension must have length 1 or 3, in ' +
                                self.name)

            if shape[2] == 3:
                self.col = 1
            else:
                self.col = 0
        elif dims != 2:
            self.col = 0
            raise Exception('Image must have 2 or 3 dimensions in ' +
                            self.name)

    def proc(self, inputs):
        o = LayerBase.proc(self, inputs)
        self.node = o
        self._request_node(o)
        return o

    def run(self):
        if self.t%eval(self._variables['Frequency']) == 0:
            img = self._get_node(self.node)
            plt.figure(self.fig_num)
            plt.clf
            if self._variables['vmin'] == '':
                vmin = img.min()
            else:
                vmin = eval(self._variables['vmin'])
            
            if self._variables['vmax'] == '':
                vmax = img.max()
            else:
                vmax = eval(self._variables['vmax'])

            plt.imshow(img, vmin = vmin, vmax = vmax)
            plt.draw()
            plt.pause(0.01)

        self.t += 1 

    def reset_real(self):
        LayerBase.reset_real(self)
        plt.figure(self.fig_num)
        plt.clf()
        self.t = 0

class Split(LayerBase):

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man,
                          output_number = 2,
                          output_names = ['a','b'])

        self._variables['dim'] = '0'
        self._variables['size of a'] = '0.5'

    def proc(self, inputs):
        pos = eval(self._variables['size of a'])
        dim = eval(self._variables['dim'])

        i = inputs['main']
        if not type(pos) is int: 
            pos = tf.floor(tf.cast(tf.shape(i)[dim],tf.float32)*pos)
            pos = tf.cast(pos, tf.int32)

        r = tf.shape(i)[dim]-pos
        pos = tf.convert_to_tensor(pos)
        sizes = tf.stack([pos, r])

        tf.assert_non_negative(r, data = [r, pos], message = "size of a must be "+
                           "less than or equal to the total size of the input")

        tf.assert_non_negative(pos, data = [r, pos], message = "size of a must be "+
                           "equal to or greater than 0")

        a, b =tf.split(i,sizes, axis = dim)
        output = {}
        output['a'] = a
        output['b'] = b
        return output

class Join(LayerBase):

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man,
                           input_number = 2,
                           input_names = ['a','b'],
                           input_share = [True, True],
                           input_shapes = [None, None],
                           input_require = [True, True])
        self._variables['dim'] = 0

    def proc(self, inputs):
        a = inputs['a']
        b = inputs['b']
        dim = eval(self._variables['dim'])
        
        SofA = self.use_variable('SizeOfA', [], init = tf.constant(0))
        
        s = tf.shape(a)[dim]

        with tf.control_dependencies([tf.assign(SofA, s)]):
            return tf.concat([a,b], dim)

class Reshape(LayerBase):

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man)
        self._variables['shape'] = '[-1]'

    def proc(self, inputs):
        i = inputs['main']
        shape = eval(self._variables['shape'])
        return tf.reshape(i, shape)

class Conv(LayerBase):

    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man,
                          input_shapes = [[None, None, None, -1]])
        self._variables['window'] = '5'
        self._variables['stride'] = '1'
        self._variables['channels'] = '10'
        self._variables['Padding'] = 'SAME'
        self._variable_types['Padding'] = VariableType('drop', ['SAME',
                                                                'VALID'])

    def proc(self, inputs):
        i = inputs['main']

        win = eval(self._variables['window'])
        stride = eval(self._variables['stride'])
        size = eval(self._variables['channels'])
        padding = self._variables['Padding']
        
        in_depth = i.get_shape().as_list()[-1]
        
        filters = self.use_variable('filters', [win, win, in_depth, size])
        biases = self.use_variable('biases', [1,1,1,size])

        tf.add_to_collection('weights', filters)
        tf.add_to_collection('biases', biases)

        result = tf.nn.conv2d(i,filters,[1, stride, stride, 1], padding)+biases 
        return result

class MaxPool(LayerBase):
    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man)
        self._variables['window'] = '2'
        self._variables['stride'] = '1'
        self._variables['Padding'] = 'SAME'
        self._variable_types['Padding'] = VariableType('drop',
                                                       ['SAME', 'VALID'])
    

    def proc(self, inputs):
        i = inputs['main']
        w = eval(self._variables['window'])
        s = eval(self._variables['stride'])
        p = self._variables['Padding']

        return tf.nn.max_pool(i, [1,w,w,1], [1,s,s,1], p)

class Softmax(LayerBase):
    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man,
                           input_number = 2,
                           input_names = ['labels','logits'],
                           input_shapes = [[None],[None, -1]],
                           input_share = [True, True],
                           input_require = [True, True])

    def proc(self, inputs):
        labels = tf.cast(inputs['labels'], tf.int32)
        logits = inputs['logits']
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,
                                                       logits = logits)
        mean_loss = tf.reduce_mean(loss)
        return mean_loss

class Shape(LayerBase):

    def proc(self, inputs):
        return tf.shape(inputs['main'])


class L2_Loss(LayerBase):
    def __init__(self, name, canvas, man):
        LayerBase.__init__(self, name, canvas, man, input_number = 0)
        self._variables['lambda'] = '4e-5'
        
        self.priority = -1

    def proc(self, inputs):
        weights = tf.get_collection('weights')
        l2s = [tf.nn.l2_loss(w) for w in weights]
        
        lam = eval(self._variables['lambda'])
        return lam*tf.add_n(l2s)

def add_layer(Class, man):
    name  = Class.__name__
    i = 0
    new_name = '%s%i'%(name, i)
    if name in list(man.names):
        while new_name in list(man.names):
            i+=1
            new_name = '%s%i'%(name, i)

        name = new_name
        
    try: 
        new_layer = Class(name, man.canvas, man)
    except TypeError as inst:
        print(inst)
        new_layer = Class(name, man.canvas)

    man.id2obj[new_layer.id] = new_layer
    man.obj2id[new_layer] = new_layer.id
    man.names.append(new_layer.name)
    man.layers.append(new_layer)
    
    return new_layer


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
        self.figs = 0
        self.shortcut_dict = {'l':Linear,
                              'c':Constant,
                              'r':Relu,
                              'p':Print,
                              'm':MSELoss,
                              'f':FileReader,
                              't':Trainer,
                              'o':Plot,
                              'e':End,
                              's':Sample,
                              'x':ExtractParam,
                              'i':ShowImage}
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
            
            tf.reset_default_graph()

            self.sess.close()
            self.sess = tf.Session()
            #as the graph is being cleared we can't keep anything from
            #previous realifying so reset. Reseting the obj will reset
            #everything in the same connected region as the obj
            obj.reset_real()
            if obj.make_real() == 0:
                print("realifying failed")
                return 
            init = tf.global_variables_initializer()
            tf.get_default_graph().finalize()
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
    














