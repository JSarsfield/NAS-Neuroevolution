"""
TensorFlow Keras model implementation of CPPN

CPPN is a biologically inspired genetic encoding/genome that produces neural network architectures when decoded.

See paper: Compositional pattern producing networks: A novel abstraction of development by Kenneth O. Stanley

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy  # deep copy genes
import operator # sort node genes by depth
import random  # random uniform weight
from genes import GeneLink


class CPPNGenome:
    """ CPPN genome - can express/decode to produce an ANN """

    def __init__(self, gene_nodes_in, gene_nodes, gene_links, num_inputs=4, num_outputs=1, var_thresh=0.01, band_thresh=0.01):
        """ Call on master thread then call a create graph function on the worker thread """
        self.weights = None  # Weight of links in graph. Sampled from parent/s genome/s or uniform distribution when no parent
        self.gene_nodes = copy.deepcopy(gene_nodes)
        self.gene_nodes_in = copy.deepcopy(gene_nodes_in)
        self.gene_links = []
        self.var_thresh = var_thresh
        self.band_thresh = band_thresh
        # Deepcopy links
        for link in gene_links:
            self.gene_links.append(GeneLink(link.weight,
                                            link.enabled,
                                            self.get_node_from_hist_marker(link.in_node.historical_marker),
                                            self.get_node_from_hist_marker(link.out_node.historical_marker),
                                            link.historical_marker))
        self.gene_nodes.sort(key=operator.attrgetter('depth'))
        node_ind = 0
        for node in self.gene_nodes_in:
            node.node_ind = node_ind
            node_ind += 1
        for node in self.gene_nodes:
            node.node_ind = node_ind
            node_ind += 1
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph = None  # Store TensorFlow graph. Created on worker thread within a create graph function

    def get_node_from_hist_marker(self, hist_marker):
        for node in self.gene_nodes:
            if node.historical_marker == hist_marker:
                return node
        for node in self.gene_nodes_in:
            if node.historical_marker == hist_marker:
                return node
        raise Exception("No node with historical marker found in func get_node_from_hist_marker genome.py")

    def create_initial_graph(self):
        """ Create an initial graph for generation zero that has no parent/s. Call on worker thread """
        # TODO mutate first, structural mutation?
        self.var_thresh = 0.01
        self.band_thresh = 0.01
        for link in self.gene_links:
            link.weight = random.uniform(-1, 1)
        self.graph = CPPNGenome.Graph(self)

    def _create_graph(self, parent_genome):
        """ Create new graph given single parent genome. Call on worker thread """
        pass  # self.genes

    def _create_graph_from_parents(self, parent_genome1, parent_genome2):
        """ Create new graph given two parent genomes.  Call on worker thread """
        pass  # self.genes

    def _perturb_weights(self):
        """ Modify the weights of the links within the CPPN """
        pass

    def visualise_genome(self):
        pass

    class Graph(nn.Module):
        """ computational graph """

        def __init__(self, genome):
            super().__init__()
            self.genome = genome
            self.weights = []  # torch tensor weights for each node
            self.activs = []  # torch activation funcs for each node
            self.outputs = torch.tensor((), dtype=torch.float32).new_empty((len(genome.gene_nodes) + genome.num_inputs))
            self.output_inds = []  # Store node indices to get output of nodes going into this node
            # Setup torch tensors
            for node in genome.gene_nodes:
                node_weights = []
                in_node_inds = []
                for link in node.ingoing_links:
                    node_weights.append(link.weight)
                    in_node_inds.append(link.out_node.node_ind)
                self.output_inds.append(torch.tensor(in_node_inds))
                self.weights.append(torch.tensor(node_weights, dtype=torch.float32))
                self.activs.append(node.act_func)

        def forward(self, x):
            """ Query the CPPN """
            n_inputs = len(x)
            # Update outputs vector with inputs
            self.outputs[torch.arange(n_inputs)] = torch.tensor(x, dtype=torch.float32)
            # loop each node and calculate output
            for i in range(len(self.activs)):
                y_unactiv = torch.dot(self.outputs[self.output_inds[i]], self.weights[i])
                y = self.activs[i](y_unactiv)
                self.outputs[i + n_inputs] = y
            return self.outputs[-self.genome.num_outputs:]
    """
    class Graph:
        def __init__(self, genome):  # Pass variables outside of the graph here
            self.genome = genome

            self.weights = []
            self.nodes = []
            self.link_maps = []  #
            self.node_locs = []  # location of nodes that perform function e.g. non input nodes
            self.paddings = []  # Zero paddings (SparseTensor doesn't have an update method yet)
            self.graph_cols = self.genome.num_inputs  # layer with most nodes
            self.graph_rows = 1  # number of layers
            self.nodes_per_row = [self.genome.num_inputs]  # number of nodes in each row/layer
            last_row_depth = 0
            nodes_in_row = 1
            x = 0
            y = 0
            for i, node in enumerate(self.genome.geneNodesIn):
                node.location = [0, i]
            # Setup tensorflow constants e.g. activation funcs & weights (CPPN weights only change during crossover)
            for node in self.genome.geneNodes:
                if node.depth != last_row_depth:
                    self.nodes_per_row.append(nodes_in_row)
                    self.graph_rows += 1
                    last_row_depth = node.depth
                    nodes_in_row = 1
                    y += 1
                    x = 0
                else:
                    nodes_in_row += 1
                    if nodes_in_row > self.graph_cols:
                        self.graph_cols = nodes_in_row
                    x += 1
                node.location = [y, x]
                self.node_locs.append(tf.constant(node.location, dtype=tf.int32, name="node_loc"))
                self.weights.append(tf.constant(
                    np.fromiter((link.weight for link in node.ingoing_links), np.float32, len(node.ingoing_links)),
                    tf.float32))
                self.nodes.append(node.activation_func)
                node_links = []
                for in_links in node.ingoing_links:
                    node_links.append(in_links.out_node.location)
                self.link_maps.append(tf.constant(node_links, dtype=tf.int32, name="links"))
            # Calculate zero paddings
            for n in self.nodes_per_row:
                self.paddings.append(
                    tf.constant(np.zeros(self.graph_cols - n, dtype=np.float32), dtype=tf.float32, name="padding"))
            
        #@tf.function
        def query(self, x1, y1, x2, y2):  # input is a Tensor with x1, x2, y1, y2

            input = np.array([x1, y1, x2, y2])
            activs = tf.zeros((self.graph_rows, self.graph_cols), dtype=tf.float32, name="activs_A")
            row0 = tf.constant(np.column_stack((np.full(input.shape[-1], 0), np.arange(input.shape[-1]))), dtype=tf.int32)
            activs = tf.tensor_scatter_nd_update(activs, row0, input, name="activs_B")
            layer_ind = 1
            activs_layer = []
            for i, node in enumerate(self.nodes):
                x = tf.gather_nd(activs, self.link_maps[i], name="x")
                activs_layer.append(node(tf.tensordot(x, self.weights[i], axes=1), name="node_out"))
                # if next node is on new layer then update activs of this layer
                if i < len(self.nodes)-2 and self.node_locs[i][0] != self.node_locs[i + 1][0] or i == len(self.nodes)-1:
                    activs = tf.tensor_scatter_nd_update(activs, list(zip(np.full(len(activs_layer), layer_ind), np.arange(len(activs_layer)))), activs_layer, name="activs_C")
                    activs_layer = []
                    layer_ind += 1
            return activs[-1, 0:self.nodes_per_row[-1]].numpy()  # tf.constant([1], dtype=tf.float32)  # tf.gather_nd(activs, [self.node_locs[-1]])
    """