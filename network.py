""" Create Artifcial Neural Network

Artificial Neural Network / Phenotype. Expressed given a genome.

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Network:
    # TODO determine how the neural network connectome is going to be represented
    # TODO determine how activation func of nodes is going to be determined

    def __init__(self, genome, links, nodes, n_net_inputs, n_net_outputs):
        self.genome = genome  # Genome used to express ANN
        self.links = links
        self.nodes = nodes
        self.nodes.sort(key=lambda node: node.y)  # Sort nodes by layer
        del self.nodes[:n_net_inputs] # Remove input nodes
        self.n_net_inputs = n_net_inputs
        self.n_net_outputs = n_net_outputs
        self.score = None  # Score the network after evaluating during lifetime
        self.weights = []  # torch tensor weights for each node
        self.activs = []  # torch activation funcs for each node
        # Setup torch tensors
        for node in self.nodes:
            # TODO!!!! we need to determine the activation function for each node from the genome
            node_weights = []
            for link in node.ingoing_links:
                node_weights.append(link.weight)
            self.weights.append(torch.tensor(node_weights, dtype=torch.float32))
            self.activs.append(torch.sigmoid)

    def create_graph(self):
        pass

    class Graph(nn.Module):
        """ computational graph """

        def __init__(self):
            super(self).__init__()

        def forward(self, x):
            """ feedforward activation of graph and return output """
            torch.dot(a, b)



class Link:
    """ Connection between two nodes """

    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight
        self.outgoing_node = None  # outgoing node
        self.ingoing_node = None  # ingoing node


class Node:

    def __init__(self, x, y, act_func=F.sigmoid):
        self.x = x
        self.y = y
        self.layer = None  # Layer number
        self.act_func = act_func
        self.ingoing_links = []  # links going into the node
        self.outgoing_links = []  # links going out of the node