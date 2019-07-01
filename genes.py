"""
Create genes e.g. connections/nodes

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import activations
import random


class GenePool:
    """
    Handles the creation and mutation of all CPPN genes throughout the evolutionary process
    """

    geneNodes = []  # Store all node genes
    geneConns = []  # Store all connection genes
    _hist_marking_num = -1  # Keeps track of historical marking number
    activation_functions = activations.ActivationFunctionSet()
    num_inputs = None

    def __init__(self,
                 num_inputs,
                 load_genepool=False):
        self.num_inputs = num_inputs
        if load_genepool is False:
            self.create_in_out_genes()

    def create_in_out_genes(self):
        """ create initial in out genes for minimal graph """
        # Create input nodes with no activation function
        for i in range(self.num_inputs):
            self.create_gene_node({"depth": 0, "activation_func": None})
        # Create output sigmoid node that provides a weight from 0 to 1
        self.create_gene_node({"depth": 1, "activation_func": activations.sigmoid_activation})
        # Add a single initial connection for each input node
        for i in range(self.num_inputs):
            self.create_gene_conn({"weight": random.uniform(-1, 1), "enabled": True, "in_node": self.geneNodes[-1], "out_node": self.geneNodes[i]})

    def create_minimal_graphs(self, n):
        """ initial generation of n minimal CPPN graphs with random weights
        Minimally connected graph with no hidden nodes, each input and output nodes should have at least one connection.
        Connections can only go forwards.
        """

        for i in range(n):
            act_func = self.activation_functions.get_random_activation_func()

    def create_gene_node(self, gene_config):
        """ Create a gene e.g. connection or node
        Must have a historical marking (array index in this case) required for crossover of parents
        """
        self.geneNodes.append(GeneNode(**gene_config))

    def create_gene_conn(self, gene_config):
        self.geneConns.append(GeneConnection(**gene_config))

    def get_new_hist_marking(self):
        self._hist_marking_num += 1
        return self._hist_marking_num

    def mutate_add_node(self):
        # Add hidden node
        pass

    def mutate_add_connection(self):
        if random.uniform(0, 1) < 0.5:
            # Split existing connection
            pass
        else:
            # Try and find two neurons to connect
            pass


class Gene:

    historical_marking = None
    can_disable = None

    def __init__(self, can_disable=True):
        self.historical_marking = get_new_hist_marking()
        self.can_disable = can_disable


class GeneConnection(Gene):

    weight = None
    enabled = None
    in_node = None
    out_node = None

    def __init__(self, weight, enabled, in_node, out_node):
        super(Gene, self).__init__()
        self.weight = weight
        self.enabled = enabled
        self.in_node = in_node
        self.out_node = out_node
        in_node.add_conn(self, True)
        out_node.add_conn(self, False)


class GeneNode(Gene):

    depth = None  # Ensures CPPN connections don't go backwards i.e. DAG
    ingoing_connections = []  # Connections going into the node
    outgoing_connections = []  # Connections going out of the node
    can_disable_conn = None
    can_enable_conn = None  # True if node has at least one connection that can be enabled
    activation_func = None  # The activation function this node contains. Incoming connections are multiplied by their weights and summed before being passed to this func

    def __init__(self, depth, activation_func):
        super(Gene, self).__init__()
        self.depth = depth
        self.activation_func = activation_func

    def add_conn(self, conn, is_ingoing):
        if is_ingoing is True:
            self.ingoing_connections.append(conn)
        else:
            self.outgoing_connections.append(conn)


