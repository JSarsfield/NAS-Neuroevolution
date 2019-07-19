"""
Create genes (links & nodes) for generating CPPN graphs

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import activations
import random
from copy import deepcopy


class GenePool:
    """
    Handles the creation and mutation of all CPPN genes throughout the evolutionary process
    """

    def __init__(self,
                 num_inputs,
                 load_genepool=False):
        self.geneNodesIn = []  # Nodes that represent input and must exist for every CPPN, these cannot be modified or disabled
        self.geneNodes = []  # Store all hidden and output node genes
        self.geneLinks = []  # Store all link genes
        self._hist_marker_num = -1  # Keeps track of historical marker number
        self.activation_functions = activations.ActivationFunctionSet()
        self.num_inputs = num_inputs
        if load_genepool is False:
            self.create_initial_genes()
        print("initial genes created")

    def create_initial_genes(self):
        """ create initial input & output genes for minimal graph """
        # Create input nodes with no activation function
        for i in range(self.num_inputs):
            self.create_initial_gene_node({"depth": 0,
                                           "activation_func": None})
        # Create output sigmoid node that provides a weight from 0 to 1
        self.create_initial_gene_node({"depth": 1,
                                       "activation_func": self.activation_functions.get("sigmoid")}, is_input=False)
        # Create initial LEO gaussian hidden nodes with bias towards locality

        # Create LEO output node
        self.create_initial_gene_node({"depth": 1,
                                       "activation_func": self.activation_functions.get("step")}, is_input=False)

        # Add a single initial link for each input node
        for i in range(self.num_inputs):
            self.create_gene_link({"weight": None,
                                   "enabled": True,
                                   "in_node": self.geneNodes[0],
                                   "out_node": self.geneNodesIn[i]})

    def create_minimal_graphs(self, n):
        """ initial generation of n minimal CPPN graphs with random weights
        Minimally connected graph with no hidden nodes, each input and output nodes should have at least one link.
        Links can only go forwards.
        """

        for i in range(n):
            act_func = self.activation_functions.get_random_activation_func()

    def create_initial_gene_node(self, gene_config, is_input=True):
        """ Create input or output gene nodes, these nodes cannot be modified or disabled and are thus treated differently from hidden node"""
        gene_config["historical_marker"] = self.get_new_hist_marker()
        if is_input:
            self.geneNodesIn.append(GeneNode(**gene_config))
        else:
            self.geneNodes.append(GeneNode(**gene_config))

    def create_gene_node(self, gene_config):
        """ Create a gene e.g. link or node
        Must have a historical marker required for crossover of parents
        """
        gene_config["historical_marker"] = self.get_new_hist_marker()
        self.geneNodes.append(GeneNode(**gene_config))

    def create_gene_link(self, gene_config):
        gene_config["historical_marker"] = self.get_new_hist_marker()
        self.geneLinks.append(GeneLink(**gene_config))

    def get_new_hist_marker(self):
        self._hist_marker_num += 1
        return self._hist_marker_num

    def mutate_add_node(self):
        # Add hidden node
        pass

    def mutate_add_link(self):
        if random.uniform(0, 1) < 0.5:
            # Split existing link
            pass
        else:
            # Try and find two neurons to connect
            pass


class Gene:

    def __init__(self, historical_marker):
        # Constants
        self.historical_marker = historical_marker


class GeneLink(Gene):

    def __init__(self, weight, enabled, in_node, out_node, historical_marker):
        super().__init__(historical_marker)
        # Constants
        self.in_node = in_node
        self.out_node = out_node
        # Variables - these gene fields can change for different genomes
        self.weight = weight
        self.enabled = enabled
        in_node.add_link(self, True)
        out_node.add_link(self, False)


class GeneNode(Gene):

    def __init__(self, depth, activation_func, historical_marker):
        super().__init__(historical_marker)
        # Constants
        self.depth = depth  # Ensures CPPN links don't go backwards i.e. DAG
        # Variables - these gene fields can change for different genomes
        self.bias = None  # Each node has a bias to shift the activation function - this is inherited from the parents and mutated
        self.act_func = activation_func  # The activation function this node contains. Incoming links are multiplied by their weights and summed before being passed to this func
        self.ingoing_links = []  # links going into the node
        self.outgoing_links = []  # links going out of the node
        self.location = None  # [x, y] 2d numpy array uniquely set for each CPPNGenome, location may be different for different genomes
        self.node_ind = None  # Set differently for each genome

    def __deepcopy__(self, memo):
        """ deepcopy but exclude ingoing_links &  outgoing_links as these will be created later """
        return GeneNode(deepcopy(self.depth, memo), deepcopy(self.act_func, memo), deepcopy(self.historical_marker, memo))

    def add_link(self, link, is_ingoing):
        if is_ingoing is True:
            self.ingoing_links.append(link)
        else:
            self.outgoing_links.append(link)

    def set_loc(self, loc):
        self.location = loc


