import tensorflow as tf
import tkinter as tk
from math import floor
import random
import string
from copy import copy 

def compare_shapes(shape1, shape2):
    """
    checks if two shapes are the same given the either can contain Nones. A
    None should always be treated as being the same.

    Args:
        shape1: the first shape
        shape2: the second shape

    Returns:
        True if they are the same, false if not
    """
    
    return all([ a == b or a == None or b == None or a == -1 or b == -1 for a,b in zip(shape1,
                                                                 shape2)])

def check_compatible(shapes):
    """    
    given a list of shapes check if they are compatible
    """

    most_specific_shape = shapes[0]

    for shape in shapes:
        if not compare_shapes(most_specific_shape, shape):
            return False

        new_mss = []
        for dim1, dim2 in zip(most_specific_shape, shape):
            if dim1 == None and dim2:
                new_mss.append(dim2)

            if dim2 == None and dim1:
                new_mss.append(dim1)

            if dim2 == None and dim1 == None:
                new_mss.append(None)
            
            if dim1 and dim2:
                new_mss.append(dim1) #must be the same as the shapes are
                #compatible

        most_specific_shape = copy(new_mss)

    return True 

class LayerBase():

    """
    The base class for layers. Each layer must know where to get its inputs
    from and have an output it that other layer objects can access. The layer
    may then have a preprocessing step where the inputs it gets can be combined
    or reshaped as required. The layer then has a main processing step which is
    defined by the layer type. Layers can be wrapped.
    The final graph is produced by finding a correct ordering of the layers
    and then calling each one to produce the correct edges in the tensorflow graph.
    Each layer must keep track of whether or not it has created its operations
    in the tensorflow graph, and so whether or not its output exists in the
    graph, this should be stored in the real flag
    Some special layers may not have outputs or inputs, for example final
    layers or input from file layers (this will depend on how these things are
    implemented)
    """

    def __init__(self, name, canvas,
                  input_number =  1,
                 input_names = ['main'],
                 input_shapes = [None],
                 input_share = [True],
                 input_require = [True]):
        """
        makes the base layer, making sure the layer is not real and has no
        inputs or variable
        
        Args:
            name: the name given to the layer
            input_shape: the shape this layers input must take, must be a list,
            if a dimension may remain unknown it can be set to None. example,
            [3,5] or [None,100]
        """
        self.input_number = input_number
        self.input_names = input_names[0:input_number]
        self.input_shapes = input_shapes[0:input_number]
        self.input_share = input_share[0:input_number]
        self.input_require = input_require[0:input_number]
        if input_number > 0:
            if len(input_names) != input_number:
                raise ValueError("number of input names must match" +
                                     " required number of inputs")
            if len(input_shapes) != input_number:
                raise ValueError("number of input shapes must match" +
                                    " required number of inputs")
            if len(input_share) != input_number:
                raise ValueError("number of input shares must match" +
                                    " required number of inputs")
            if len(input_require) != input_number:
                raise ValueError("number of input require must match" +
                                    " required number of inputs")
        empties = [ [] for _ in range(input_number)] 
        self._inputs = dict(zip(input_names, empties))
        self._making_real = False
        self._variables = {}
        self.real = False
        self._outputs = []
        self.output = None
        self.name = name
        self._canvas = canvas
        self.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))

        self._resetting = False  
        
        self.repeat = 1

        self.arrows = {}
        self.print_output = False
        self.log_output = False
       
        self._tf_vars = []

        self.x0 = 10
        self.y0 = 10
        self.x1 = 120
        self.y1 = 90
        self._v_win_width = 250
        self._v_win_height = 40
        
        self.showing_v_win = False

        self.input_height = 20
        self.shape, self.text, self.sockets = self._make_shapes()

    @property
    def x_mean(self):
        return (self.x0 + self.x1)/2

    @property
    def y_mean(self):
        return (self.y1 + self.y0)/2


    @property
    def coords(self):
        id_search = self._canvas.find_withtag(self.id)
        body_search = self._canvas.find_withtag('body')

        if len(id_search) > len(body_search):
            smaller = body_search
            larger = id_search
        else:
            smaller = id_search
            larger = body_search

        for i in smaller:
            if i in larger:
                return self._canvas.coords(i)
        

    def _make_shapes(self):
        sockets = {}
        #alias
        r = self._canvas.create_rectangle
        t = self._canvas.create_text 
        if self.input_number: 
            #shrink main box
            self.y1 -= self.input_height
            
            #find socket widths
            i_width = floor((self.x1 - self.x0)/self.input_number)
            remainder = (self.x1 - self.x0)%self.input_number

            widths = remainder*[i_width+1] + (self.input_number-remainder)*[i_width]
            
            #find socket coords
            x0s = [self.x0 + sum(widths[:i]) for i in range(self.input_number)]
            x1s = [x0s[i] + widths[i] for i in range(self.input_number)]
            y0 = self.y1
            y1 = self.y1 + self.input_height

            #make each socket
            for i,n in zip(range(self.input_number), self.input_names):
                socket_box = r(x0s[i], y0, x1s[i], y1,tags = (self.id, 'input', n))
                socket_text = t((x0s[i]+x1s[i])/2, (y0+y1)/2, text = n, 
                                tags = (self.id,'input', n,  'socket_name'))
                socket = [socket_box, socket_text]
                sockets[n] = socket


        shape = r(self.x0,self.y0,self.x1,self.y1, tags = (self.id, 'body'))

        text = t(self.x_mean, self.y_mean, text = self.name, tags=(self.id, 'name'))
            
        return[shape, text, sockets]

    def check_inputs(self, inputs):
        """
        checks if the inputs are compatible 
    
        Args:
            inputs: a list of tensors
    
        Raises:
            ValueError if too many inputs are given
                       or if the inputs have inconsisten dimensions
                       or if the inputs shapes do not match the required shape 
        """
        for n,share,shape, req in zip(self.input_names, self.input_share, self.input_shapes,
                                      self.input_require):
            i = self._inputs[n]
            
            nme = self.name + ', ' + n 
            
            if (not share) and len(i)>1:
                raise Exception("More than one input to non-shared socket in "
                                + nme)

            if req and len(i) == 0:
                raise Exception("No input to socket which requires input in " +
                                nme)
            if shape:
                i_shapes = [shape] + [t.output.get_shape().as_list() for t in i]
            else:
                d = len(i[0].output.get_shape().as_list())
                if d > 0:
                    i_shapes = [(d-1)*[None] + [-1]] +  [t.output.get_shape().as_list() for t in i]
                else:
                    i_shapes = [t.output.get_shape().as_list() for t in i]

            dims = [len(s) for s in i_shapes]

            if not all([d == dims[0] for d in dims]):
                raise Exception("inconsistent dimensions in " + nme)

            if not check_compatible(i_shapes):
                raise Exception("incompatible shapes in " + nme)
            
        
    def pre_proc(self, inputs):
        """
        the pre processing step, this base class is an identity leaving the
        data unchanged but concatenated.
        Args:
            inputs: a dictionary of lists of compatible tensors

        Returns:
            the input if only a single input is allowed into this layer or the
            inputs concatenated along the final dimension (as this is generally
            the depth) if more than one input is allowed
        """
    
        outputs = {}
        with tf.variable_scope(self.name):
            for n in self.input_names:
                dim = len(inputs[n][0].get_shape().as_list())
                outputs[n] = tf.concat(inputs[n], dim-1)
        return outputs 
 

    def proc(self, inputs):
        """
         processes a dictionary of tensors, happens after pre_proc

        Args:
            inputs: dictionary of tensors

            returns: the layer output
        """
        if self.input_number == 1:
            n = self.input_names[0]
            with tf.variable_scope(self.name):
                return inputs[n]

    def make_real(self):
        """
        if this function is called while _making_real is true then it must have, through
        some other layers or directly, called itself, implying a loop in the
        diagram, so raise an exception.
        
        """
        if (not self.real) and self._making_real:
            raise Exception ('Recusion Error, check for loops')
        
        if not self.real:
            self._making_real = True
            
            for n in self.input_names:
                for i in self._inputs[n]:

                    if i.make_real() == 0:
                        #something went wrong
                        return 0
                #recursively make everything real, the base of
                #the recursion tree will be at layers with not inputs, these
                #will be layers which read files or output a constant
            for n in self.input_names:
                if not all([i.real for i in self._inputs[n]]):
                    raise Exception('Not all inputs have real outputs')
             
            inputs = {}
            for n in self.input_names:
                inputs[n] = [i.output for i in self._inputs[n]]

            try:
                self.check_inputs(inputs) #will raise errors if there is a problem
            except Exception as inst:
                self.set_real(False)
                self._making_real = False
                print(inst)
                return 0


            self._inter = self.pre_proc(inputs)

            self.output = self.proc(self._inter)
             
            self.set_real(True)
            self._making_real = False
            return 1

    def run(self, session):
        return

    @property
    def num_inputs(self):
        return len(self._inputs)

    def add_input(self, new_input, socket):
        """
        adds a new input to the layer.
        Args:
            new_input: the layer that the new input is coming from
        """
        if not isinstance(new_input, LayerBase):
            raise ValueError('new_input must be another layer')
        
        if new_input in self._inputs[socket]:
            return 
        
        self.reset_real()
        self._inputs[socket].append(new_input)
        

        X = self._canvas.coords(self.sockets[socket][0])
        x1 = (X[0]+X[2])/2
        y1 = X[3]
        
        X1 = self._canvas.coords(new_input.shape)
        x0 = (X1[0]+X1[2])/2
        y0 = X1[1]
        
        id1 = 'e' + self.id
        id2 = 's' + new_input.id 

        self.arrows[(new_input,socket)] = (self._canvas.create_line(x0,y0,x1,y1,
                                                    arrow = 'last',
                                                    tags = (id1, id2,'arrow')))
        new_input._outputs.append(self)  

    def remove_input(self, old_input, socket):
        if old_input not in self._inputs[socket]:
            return 
        
        self.reset_real()
        self._canvas.delete(self.arrows[(old_input,socket)])
        del self.arrows[(old_input,socket)]

        self._inputs[socket].remove(old_input)
        old_input._outputs.remove(self)

    def change_name(self, new_name):
        if self.real:
            raise Exception('cannot change the name of a real layer, graph' +
                            'must be cleared before making changes')
        self.name = new_name
        self.reset_real()
    def reset_real(self):
        
        self._resetting = True

        self.set_real(False)
        self._making_real = False 
        
        for o in self._outputs:
            if not o._resetting:
                o.reset_real()
        
        for v in self._inputs.values():
            for i in v:
                if not i._resetting:
                    i.reset_real()
 
        self._resetting = False 

    def set_real(self, r):
        if r:
            self._canvas.itemconfig(self.shape, fill = 'green')
        else:
            self._canvas.itemconfig(self.shape, fill = '')
        self.real = r
        
    def show_variable_window(self):
     
        if len(self._variables) == 0:
            return
        
        self.showing_v_win = True

        X = self.coords
        num = len(self._variables)
        x0 = X[2]
        x1 = x0 + self._v_win_width
        y0 = X[1]
        y1 = y0 + self._v_win_height*num
        
        self._v_win = self._canvas.create_rectangle(x0,y0,x1,y1, tags = self.id)
        self._v_win_labels = []
        self._v_win_entries = []
        vs = self._variables.values()
        ns = self._variables.keys()
        pad = 5
        for v,n,i in zip(vs, ns, range(num)):
            self._v_win_labels.append(self._canvas.create_text(x0 + pad,
                                                               y0 + (i+1)*pad,
                                                               anchor = tk.NW,
                                                               text = n,
                                                               tags = self.id))

            sv = tk.StringVar()

            def callback(sv):
                try:
                    new_val = sv.get()     
                except:
                    return
                if new_val != self._variables[n]:
                    self._variables[n] = new_val
                    self.reset_real()

            sv.trace("w", lambda name, index, mode, sv=sv: callback(sv))
            e = tk.Entry(self._canvas, textvariable = sv)
            e.config(state = 'normal')
            e.insert(0,str(v))
            w = self._canvas.create_window(x0 + 50, y0 + (i+1)*pad, window = e,
                                           anchor = tk.NW, tags = self.id)
            self._v_win_entries.append((e,w)) 

    def hide_variable_window(self):
        if not self.showing_v_win:
            return
    

        self._canvas.delete(self._v_win)
        self._v_win = None 
        
        for s in self._v_win_labels:
            self._canvas.delete(s)
        
        self._v_win_labels = []
        
        for s in self._v_win_entries:
            s[0].config(state = 'disabled')
            self._canvas.delete(s[1])

        self._v_win_entries = []

        self.showing_v_win = False
        









