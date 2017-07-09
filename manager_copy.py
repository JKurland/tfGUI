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
                            'i':ShowImage}

        self.figs = 0
        self.session = tf.Session()

        self.id2obj = {}

        self._selected = {}
        self._selected_arr = []

        self._short = 5

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
        
        close = self.canvas.find_overlapping(x-range, y-range, x+range, y+range)
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
        self._events = filter(lambda x: x[1] != e_type, self._events)
         

    def _get_event_type(self, e_type):
        y = filter(lambda x: x[1] == e_type, self._events)
        return y

    def _select(self, block):
        if self._select[block] == 0:
            block.select()
            self._selected[block] = 1
            self._selected_arr.append(block) 

    def _deselect(self, block):
        if self._selected[block] == 1:
            block.deselect()
            self._selected[block] = 0
            self._selected_arr.remove(block)

    def reset_graph(self):
        tf.reset_default_graph
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
            block.make_real()
        
        self.h = self.session.partial_run_setup(block._partial_run_nodes,[])
         
        for b in self._blocks:
            if b.real:
                b.run


    def on_click(self, event):
        self._add_event(event, 'click') 
        self.x = event.x
        self.y = event.y
    
    def on_release(self, event):
        
        c_event, c_type, c_block, c_tags = self._get_event_type('click')[-1]
        block, tags = self._find_block(event)
      
        distance = self._event_distance(c_event, event)

        self._remove_event_type('click')
 
        if c_block == None and block == None and distance < self._short:
            self._clear_selection() 
        elif 'arrow' in tags:
            return
        elif c_block == None or block == None:
            return 
        elif block is c_block and distance < self._short:
           self._select(block)
        else:
             return

    def on_move(self, event):
 
        c_event, c_type, c_block, c_tags = self._get_event_type('click')[-1]
        block, tags = self._find_block(event)
      
        distance = self._event_distance(c_event, event)
        
        if distance < self._short:
            return
        elif block == None or c_block == None:
            return
        elif block is c_block:
            block.move_by(event.x - self.x, event.y - self.y)
            self.x = event.x
            self.y = event.y
        else:
            return 

    def on_r_click(self, event):
        self._add_event(event, 'r_click')
          

    def on_r_release(self, event):
         
        c_event, c_type, c_block, c_tags = self._get_event_type('r_click')[-1]
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
            c_block.toggle_input(block, out_socket, in_socket)
            #need to add this to the state for saving
        else:
            return

    def on_double(self, event):
        block, tags = self._find_block(event)

        if block == None:
            return
        elif block.real:
            for b in self._blocks:
                b.run()
        else:
            block.make_real()
            block.run()

    def on_key(self, event):
        
        if len(self._selected_arr) > 0:
            if event.char == 'v':
                for block in self._selected_arr:
                    block.toggle_variables()
        elif event.char in self.shortcut_dict:
            self._add_block(self.shortcut_dict[event.char])
        else:
            return









