a graphical interface for creating tensorflow graphs and running them

## Structure

Each type of block is represented by a class. To specify the behabiour of a
block there are a number of base class functions that can be overridden.

pre_proc(self, inputs)   
proc(self, inputs)   
run(self, session)  

pre_proc is called first, when the graph is being built, and by default concatenates all the shared inputs
along the last dimension. For example if more than one input is given to a
Linear block the inputs will be combined depthwise. This requires that the
batch-wise dimensions match.

proc is called next, the output of pre_proc is fed directly into the inputs of
proc. The tensor returned by proc is the output of the block. By default this
function does nothing but return inputs['main']

run is called when the block is called, either by being double clicked on or
when the block above it is called. Session is the current tensorflow session.

##sockets

each block can have many different named inputs called sockets. by default each
block has one socket called 'main'. This means that the inputs passed to both
proc and pre_proc are dictionaries with a single key, value pair, with the key
being 'main'.



At the moment there are three files which really do anything, main, simple
layers and layer base. The first runs the program.

## Simple layers
contains class definitions for each block and the drag_manager class which
deals with all the GUI input at the moment

## Layer base
contains the base class definition, all block classes inherit from this class.
All the defaults are defined here as well as the machinery for building the
graph
