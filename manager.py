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

        self._short = 5

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

    def _deselect(self, block):
        if self._selected[block] == 1:
            block.deselect()
            self._selected[block] = 0

    def _clear_selection(self):
        for block in self._selected:
            block.deselect()
            self._deselect(block)

    def on_click(self, event):
        self._add_event(event, 'click') 
 
    
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
            block.move_to(event.x, event.y)
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
            c_block.add_input(block, out_socket, in_socket)
            #need to add this to the state for saving
        else:
            return












