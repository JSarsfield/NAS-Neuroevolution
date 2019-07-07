"""
Create genes (links & nodes) for generating CPPN graphs

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import activations
import random


class GenePool:
    """
    Handles the creation and mutation of all CPPN genes throughout the evolutionary process
    """

    def __init__(self,
                 num_inputs,
                 load_genepool=False):
        self.geneNodesIn = []  # Nodes that represent input and must exist for every CPPN, these cannot be modified or disabled
        self.geneNodesOut = []  # Nodes that represent output and must exist for every CPPN, these cannot be modified or disabled TODO allow modification of activation func
        self.geneNodes = []  # Store all hidden node genes
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
        # Add a single initial link for each input node
        for i in range(self.num_inputs):
            self.create_gene_link({"weight": random.uniform(-1, 1),
                                   "enabled": True,
                                   "in_node": self.geneNodesOut[0],
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
            self.geneNodesOut.append(GeneNode(**gene_config))

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
        self.historical_marker = historical_marker


class GeneLink(Gene):

    def __init__(self, weight, enabled, in_node, out_node, historical_marker):
        super().__init__(historical_marker)
        self.weight = weight
        self.enabled = enabled
        self.in_node = in_node
        self.out_node = out_node
        in_node.add_link(self, True)
        out_node.add_link(self, False)


class GeneNode(Gene):

    def __init__(self, depth, activation_func, historical_marker):
        super().__init__(historical_marker)
        self.depth = depth  # Ensures CPPN links don't go backwards i.e. DAG
        self.activation_func = activation_func  # The activation function this node contains. Incoming links are multiplied by their weights and summed before being passed to this func
        self.ingoing_links = []  # links going into the node
        self.outgoing_links = []  # links going out of the node
        self.can_disable_link = None
        self.can_enable_link = None

    def add_link(self, link, is_ingoing):
        if is_ingoing is True:
            self.ingoing_links.append(link)
        else:
            self.outgoing_links.append(link)


