from simple_layers import *
import tkinter as tk
from math import sqrt


class Manager():

    def __init__(self, canvas):
   
        self._canvas = canvas

        self._events = []
        
        self._blocks = []

        self._state = {}

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
                            'i':ShowImage,
                            'k':Split,
                            'h':Reshape,
                            'j':Join,
                            'n':Conv,
                            'a':MaxPool,
                            'y':Softmax,
                            'g':Shape,
                            '2':L2_Loss}

        self.figs = 0
        self.session = tf.Session()

        self.id2obj = {}

        self._selected = {}
        self._selected_arr = []

        self._short = 5
        
        self._bind_events()

    def _bind_events(self):
        self._canvas.bind("<Button-1>", self.on_click)
        self._canvas.bind("<B1-Motion>", self.on_move)
        self._canvas.bind("<ButtonRelease-1>", self.on_release)   
        self._canvas.bind_all("<Button-3>", self.on_r_click)
        self._canvas.bind_all("<ButtonRelease-3>", self.on_r_release)
        self._canvas.bind_all("<KeyPress>", self.on_key)
        self._canvas.bind("<Double-Button-1>", self.on_double)

    def _add_block(self, Class):
        base_name = Class.__name__
        i = 0
        names = [block.name for block in self._blocks]
        
        name = base_name 
        while name in names:
            name = base_name + '%i'%i
            i += 1

        new_block = Class(name, self._canvas, self)

        self._blocks.append(new_block)
        self._selected[new_block] = 0
        self.id2obj[new_block.id] = new_block

    def _event_distance(self, event1, event2):
        x1 = event1.x
        x2 = event2.x

        y1 = event1.y
        y2 = event2.y

        dist = sqrt( (x1 - x2)**2 + (y1 - y2)**2)
        return dist

    def _find_close(self, x, y, range = 1):
        
        close = self._canvas.find_overlapping(x-range, y-range, x+range, y+range)
        if len(close) == 0:
            return [None]
        else:
            return close

    def _find_block(self, event):
        shape = self._find_close(event.x, event.y)[0]
        if shape:
            tags = self._canvas.gettags(shape)
            if 'arrow' not in tags:
                block = self.id2obj[tags[0]]
            else:
                block = None
        else:
            block = None
            tags = None
        
        return block, tags

    def _add_event(self, event, e_type):
        block, tags = self._find_block(event)
        
        self._events.append([event, e_type, block, tags])

    def _remove_event_type(self, e_type):
        self._events = list(filter(lambda x: x[1] != e_type, self._events))
         

    def _get_event_type(self, e_type):
        y = list(filter(lambda x: x[1] == e_type, self._events))
        return y

    def _select(self, block):
        
        if self._selected[block] == 0:
            self._clear_selection()
            block.select()
            self._selected[block] = 1
            self._selected_arr.append(block) 

    def _deselect(self, block):
        if self._selected[block] == 1:
            block.hide_variable_window()
            block.deselect()
            self._selected[block] = 0
            self._selected_arr.remove(block)

    def reset_graph(self):
        tf.reset_default_graph()
        self.session.close()
        self.session = tf.Session()
        for block in self._blocks:
            block.reset()

    def _clear_selection(self):
        for block in self._selected:
            block.deselect()
            self._deselect(block)

    def _run_block(self, block):
        if not block.real:
            if block.make_real() == 0:
                return
            self.init = tf.global_variables_initializer()
            self.session.run(self.init)

        blocks_to_run = list(block.dependencies)
        p_nodes = list(block._partial_run_nodes)

        if len(p_nodes) > 0:
            self.h = self.session.partial_run_setup(p_nodes,[])
        else:
            self.h = None

        for b in blocks_to_run:
            if b.real:
                b.run() 
        while block.cont:
            if len(p_nodes) > 0:
                self.h = self.session.partial_run_setup(p_nodes,[])
            else:
                self.h = None
            for b in blocks_to_run:
                if b.real:
                    b.run()

    def _remove_block(self, block):
        block.clear_connections()
        block.delete_shapes()
        self._blocks.remove(block)
        
        self._deselect(block)
        block.hide_variable_window()
        
        try:
            self._selected_arr.remove(block)
        except:
            pass
        
        try:
            self.id2obj.pop(block.id)
        except:
            pass

        try:
            self._selected.pop(block)
        except:
            pass

        del block

    def on_click(self, event):
        
        self._add_event(event, 'click') 
        self.x = event.x
        self.y = event.y
    
    def on_release(self, event):
        
        l = self._get_event_type('click')
        if len(l)>0:
            c_event, c_type, c_block, c_tags = l[-1]
        else:
            return #double click release as on_click is not called
        #on the second click of a double click. This means l will have been
        #emptied and not added to againprint('r') 

        block, tags = self._find_block(event)
      
        distance = self._event_distance(c_event, event)

        self._remove_event_type('click')
 
        if c_block == None and block == None and distance < self._short:
            self._clear_selection() 
        elif c_block == None or block == None:
            return 
        elif block is c_block and distance < self._short:
            if block.selected:
                self._deselect(block)
            else:
                self._select(block)
        else:
             return

    def on_move(self, event): 
        c_event, c_type, c_block, c_tags = self._get_event_type('click')[-1]
        block, tags = self._find_block(event)
      
        distance = self._event_distance(c_event, event)
        
        if distance < self._short:
            return
        elif c_block == None:
            return
        else:
            c_block.move_by(event.x - self.x, event.y - self.y)
            self.x = event.x
            self.y = event.y
        
    def on_r_click(self, event):
        self._add_event(event, 'r_click')
          

    def on_r_release(self, event):
        
        l = self._get_event_type('r_click')
        if len(l)>0:
            c_event, c_type, c_block, c_tags = l[-1]
        else:
            return #double click release as on_click is not called
        #on the second click of a double click. This means l will have been
        #emptied and not added to again
        block, tags = self._find_block(event)
      
        distance = self._event_distance(c_event, event)

        self._remove_event_type('r_click')
        
        if block == None and c_block == None and distance < self._short:
            return #right click menu will go here
        elif block == None or c_block == None:
            return 
        elif block is c_block and distance < self._short:
            block.show_r_click_menu()
        elif 'input' in tags and 'output' in c_tags and c_block is not block:
            out_socket = c_tags[2]
            in_socket = tags[2]
            block.toggle_input(c_block, in_socket, out_socket)
            #need to add this to the state for saving
        else:
            return

    def on_double(self, event):

        block, tags = self._find_block(event)

        if block == None:
            return
        else:
            self._run_block(block)

    def on_key(self, event):

        try:
            code = ord(event.char)
        except:
            code = 0 

        if len(self._selected_arr) > 0:
            if event.char == 'v' and not any([b.showing_v_win for b in
                                              self._selected_arr]):
                for block in self._selected_arr:
                    block.toggle_variables()
            elif code == 127:
                for block in self._selected_arr:
                    self._remove_block(block)
        elif event.char in self.shortcut_dict:
            self._add_block(self.shortcut_dict[event.char])
        else:
            return









