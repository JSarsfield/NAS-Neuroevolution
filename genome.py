"""
CPPN implementation using PyTorch

CPPN is a biologically inspired genetic encoding/genome that produces neural network architectures when decoded.

See papers: 1. Compositional pattern producing networks: A novel abstraction of development by Kenneth O. Stanley
2. A hypercube-based encoding for evolving large-scale neural networks. Stanley, K., D’Ambrosio, D., & Gauci, J. (2009)

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
from activations import ActivationFunctionSet
from config import *


# TODO visualise genome

class CPPNGenome:
    """ CPPN genome - can express/decode to produce an ANN """

    def __init__(self, gene_nodes_in, gene_nodes, gene_links, num_inputs=4, num_outputs=2, var_thresh=0.01, band_thresh=0.01):
        """ Call on master thread then call a create graph function on the worker thread """
        self.weights = None  # Weight of links in graph. Sampled from parent/s genome/s or uniform distribution when no parent
        self.gene_nodes = copy.deepcopy(gene_nodes)
        self.gene_nodes_in = copy.deepcopy(gene_nodes_in)
        self.gene_links = []
        self.var_thresh = var_thresh
        self.band_thresh = band_thresh
        self.species = None  # Species this genome belongs to
        self.act_set = ActivationFunctionSet()
        # Deepcopy links
        if type(gene_links[0]) is tuple:
            for link in gene_links:
                self.gene_links.append(GeneLink(link[0],
                                                self.get_node_from_hist_marker(link[1]),
                                                self.get_node_from_hist_marker(link[2]),
                                                link[3],
                                                enabled=link[4]))
        else:
            for link in gene_links:
                self.gene_links.append(GeneLink(link.weight,
                                                self.get_node_from_hist_marker(link.in_node.historical_marker),
                                                self.get_node_from_hist_marker(link.out_node.historical_marker),
                                                link.historical_marker,
                                                enabled=link.enabled))
        self.gene_nodes.sort(key=operator.attrgetter('depth'))
        self.gene_links.sort(key=lambda x: x.historical_marker)  # Sorted genome required for speciation
        node_ind = 0
        for node in self.gene_nodes_in:
            node.node_ind = node_ind
            node_ind += 1
        for node in self.gene_nodes:
            node.node_ind = node_ind
            node_ind += 1
        self.cppn_inputs = num_inputs
        self.cppn_outputs = num_outputs
        self.net = None  # neural network expressed by this genome
        self.graph = None  # PyTorch graph.

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
        self.var_thresh = 0.3
        self.band_thresh = 0
        # Initialise weights
        for link in self.gene_links:
            link.weight = random.uniform(-1, 1)
        # Initialise biases
        for node in self.gene_nodes:
            node.bias = random.uniform(-0.1, 0.1)
            if node.can_modify:
                node.act_func = self.act_set.get_random_activation_func()
        self.graph = CPPNGenome.Graph(self)

    def create_graph(self):
        """ Create graph """
        self.graph = CPPNGenome.Graph(self)

    def mutate_nonstructural(self):
        """ perform nonstructural mutations to existing gene nodes & links """
        for link in self.gene_links:
            # Disable/Enable links
            if event(link_toggle_prob):  # Chance of toggling link
                link.enabled = True if link.enabled is False else False
            if link.enabled is False and event(link_enable_prob):  # Chance of enabling a disabled link
                link.enabled = True
            # Mutate weights
            if event(weight_mutate_rate):
                link.weight += np.random.normal(scale=gauss_weight_scale)
        for node in self.gene_nodes:
            # Mutate bias
            if event(bias_mutate_rate):
                node.bias += np.random.normal(scale=gauss_weight_scale)
            # Mutate activation func
            if event(change_act_prob):
                node.act_func = self.act_set.get_random_activation_func()

    def set_species(self, species):
        """ set the species this genome belongs to """
        self.species = species

    def visualise_genome(self):
        pass

    class Graph(nn.Module):
        """ computational graph """

        def __init__(self, genome):
            super().__init__()
            self.genome = genome
            self.weights = []  # torch tensor weights for each node
            self.node_funcs = []  # torch node funcs
            self.activs = []  # torch activation funcs for each node
            self.outputs = torch.tensor((), dtype=torch.float32).new_empty((len(genome.gene_nodes) + genome.cppn_inputs))
            self.output_inds = []  # Store node indices to get output of nodes going into this node
            self.node_biases = []
            # Setup torch tensors
            for node in genome.gene_nodes:
                node_weights = []
                in_node_inds = []
                for link in node.ingoing_links:
                    node_weights.append(link.weight)
                    in_node_inds.append(link.out_node.node_ind)
                self.output_inds.append(torch.tensor(in_node_inds))
                self.weights.append(torch.tensor(node_weights, dtype=torch.float32))
                self.node_funcs.append(node.node_func)
                self.activs.append(node.act_func)
                self.node_biases.append(node.bias)
            self.node_biases = torch.tensor(self.node_biases, dtype=torch.float32)

        def forward(self, x):
            """ Query the CPPN """
            n_inputs = len(x)
            # Update outputs vector with inputs
            self.outputs[torch.arange(n_inputs)] = torch.tensor(x, dtype=torch.float32)
            # loop each node and calculate output
            for i in range(len(self.activs)):
                y_unactiv = self.node_funcs[i](self.outputs[self.output_inds[i]], self.weights[i], self.node_biases[i])
                try:
                    y = self.activs[i](y_unactiv)
                except:
                    print("")
                self.outputs[i + n_inputs] = y
            return self.outputs[-self.genome.cppn_outputs:]