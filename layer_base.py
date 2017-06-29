import tensorflow as tf
import random
import string

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
    
    if shape1 == None or shape2 == None:
        return True

    return all([ a == b or a == None or b == None for a,b in zip(shape1,
                                                                 shape2)])

class DragManager():
    """
    class for an object which managesthe dragging of objects
    """

    def __init__(self, canvas):
        self.widget = None
        self.canvas = canvas
        self.mouse_x = None
        self.mouse_y = None 

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

    def __init__(self, name, canvas, input_shape,
                 allow_multiple_inputs = False,
                 require_input = True):
        """
        makes the base layer, making sure the layer is not real and has no
        inputs or variable
        
        Args:
            name: the name given to the layer
            input_shape: the shape this layers input must take, must be a list,
            if a dimension may remain unknown it can be set to None. example,
            [3,5] or [None,100]
        """
        self._making_real = False
        self._mult_inputs = allow_multiple_inputs
        self._variables = []
        self.real = False
        self._inputs = []
        self.output = None
        self.name = name
        self._input_shape = input_shape
        self._require_input = require_input
        self._canvas = canvas
        self.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
        self._shape = canvas.create_rectangle(10,10,120,90, tags = self.id)
        self._text = canvas.create_text(65, 50, text = self.name,
                                        tags = self.id)
        self.arrows = {}
        
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
         
        if len(self._inputs) == 0 and self._require_input:
            raise ValueError(self.name + ' requires an input but none are ' +
                             'given')

        if (not self._mult_inputs) and len(self._inputs) > 1:
            raise ValueError('only one input is allowed into ' + self.name + 
                             'but %i inputs given'%len(self._inputs))
    
    
        shapes = [i.get_shape().as_list() for i in inputs] 
        if self._input_shape == None:
            l = len(inputs[0].get_shape().as_list())
        else:
            l = len(self._input_shape)

        #all the inputs must have the same number of dimensions as input shape
        if not all([len(shape) == l for shape in shapes]):
            raise ValueError('all inputs must have the same number of ' +
                             'dimensions, error in ' + self.name)

        if not all([compare_shapes(self._input_shape, shape) for shape in
                    shapes]):
            raise ValueError('all input shapes must match the required input ' +
                             ' shape, error in ' + self.name)
         
        
    
    def pre_proc(self, inputs):
        """
        the pre processing step, this base class is an identity leaving the
        data unchanged but concatenated.
        Args:
            inputs: a list of compatible tensors

        Returns:
            the input if only a single input is allowed into this layer or the
            inputs concatenated along the final dimension (as this is generally
            the depth) if more than one input is allowed
        """
        if self._mult_inputs:   
            with tf.variable_scope(self.name):
                dim = len(inputs[0].get_shape().as_list())
                return tf.concat(inputs,dim-1)
        else:
            return inputs[0]

                    


    def proc(self, inputs):
        """
        takes a single input tensor or a list of tensors and does the main
        processing for the layer. This base layer is the identity

        Args:
            inputs: the input tensor or tensors

            returns: the layer output
        """
        with tf.variable_scope(self.name):
            return inputs

    def make_real(self):
        """
        if this function is called while _making_real is true then it must have, through
        some other layers or directly, called itself, implying a loop in the
        diagram, so raise an exception.
        
        """
        if self._making_real:
            raise Exception ('Recusion Error, check for loops')
        
        if not self.real:
            self._making_real = True

            for i in self._inputs:
                i.make_real() #recursively make everything real, the base of
                #the recursion tree will be at layers with not inputs, these
                #will be layers which read files or output a constant

            if not all([i.real for i in self._inputs]):
                raise Exception('Not all inputs have real outputs')
            
            inputs = [i.output for i in self._inputs]
            self.check_inputs(inputs) #will raise errors if there is a problem

            self._inter = self.pre_proc(inputs)

            self.output = self.proc(self._inter)
             
            self.real = True
            self._making_real = False

    
    @property
    def num_inputs(self):
        return len(self._inputs)

    def add_input(self, new_input):
        """
        adds a new input to the layer.
        Args:
            new_input: the layer that the new input is coming from
        """
        if not isinstance(new_input, LayerBase):
            raise ValueError('new_input must be another layer')
        
        if new_input in self._inputs:
            return 

        self._inputs.append(new_input)
        
        X = self._canvas.coords(self._shape)
        x1 = (X[0]+X[2])/2
        y1 = X[3]
        
        X1 = self._canvas.coords(new_input._shape)
        x0 = (X1[0]+X1[2])/2
        y0 = X1[1]
        
        id1 = 'e' + self.id
        id2 = 's' + new_input.id 

        self.arrows[new_input] = (self._canvas.create_line(x0,y0,x1,y1,
                                                    arrow = 'last',
                                                    tags = (id1, id2)))
        
    def remove_input(self, old_input):
        if old_input not in self._inputs:
            return 

        self._canvas.delete(self.arrows[old_input])
        del self.arrows[old_input]

        self._inputs.remove(old_input)

    def change_name(new_name):
        if self.real:
            raise Exception('cannot change the name of a real layer, graph' +
                            'must be cleared before making changes')
        self.name = new_name

    def reset_real(self):
        self.real = False
        self._making_real = False 
