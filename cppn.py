"""
Compositional Pattern Producing Network

A biologically inspired genetic encoding that produces neural network architectures when decoded.

See paper: Compositional pattern producing networks: A novel abstraction of development by Kenneth O. Stanley


__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import activations

genes = []  # Store genes
activation_functions = activations.ActivationFunctionSet()


def create_random_graphs(n, inputs=2):
    """ initial generation of n minimal CPPN graphs with random weights
    Minimally connected graph with no hidden nodes, each input and output nodes should have at least one connection.
    Connections can only go forwards.
    """

    for i in range(n):
        act_func = activation_functions.get_random_activation_func()

def create_gene():
    """ Create a gene e.g. connection or node
    Must have a historical marking (array index in this case) required for crossover of parents
    """

class Gene:

    historical_marking = None  # index

    def __init__(self):

class GeneConnection(Gene):

    weight = None

    def __init__(self, hist_marking):
        super(Gene, self).__init__(hist_marking)

class GeneNode(Gene):

    depth = None  # Ensures CPPN connections don't go backwards i.e. DAG

    def __init__(self, _depth, hist_marking):
        depth = _depth
        super(Gene, self).__init__(hist_marking)