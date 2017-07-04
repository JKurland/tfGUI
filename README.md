a graphical interface for creating tensorflow graphs and running them

## Structure

Each type of block is represented by a class. To specify the behabiour of a
block there are a number of base class functions that can be overridden.

pre_proc(self, inputs)   
proc(self, inputs)   


At the moment there are three files which really do anything, main, simple
layers and layer base. The first runs the program.

## Simple layers

