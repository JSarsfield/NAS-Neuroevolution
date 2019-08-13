""" Create Artifcial Neural Network

Artificial Neural Network / Phenotype. Expressed given a genome.

 NOTE self.fitness calc is slightly different from original NEAT - see adjusted fitness section 3.3 pg 12 "Evolving Neural Networks through Augmenting Topologies" 2002

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def relu(x):
    return max(0, x)


def step(x):
    return 1 if x > 0 else 0


class Network:
    # TODO determine how activation func of nodes is going to be determined
    # TODO if number of outputs is greater than 1 is there always going to be a path to all output nodes??
    # TODO GRADIENT BASED LIFETIME LEARNING - each node should have trainable bias initialised close to zero. Don't put bias nodes on output layer?

    def __init__(self, genome, links, nodes, n_net_inputs, n_net_outputs, void=False):
        self.is_void = void
        self.genome = genome  # Genome used to express ANN
        self.links = links
        self.nodes = nodes
        for i, node in enumerate(self.nodes):
            node.node_ind = i
        self.input_nodes = self.nodes[:n_net_inputs]
        del self.nodes[:n_net_inputs] # Remove input nodes
        self.n_net_inputs = n_net_inputs
        self.n_net_outputs = n_net_outputs
        self.discrete = True if n_net_outputs == 1 else False
        self.fitness_unnorm = -9999  # Un-normalised fitness of net
        self.fitness = -9999  # Fitness of net normalised for size of species
        self.genome.net = self
        if void:
            return
        self.graph = Network.Graph(self)

        # TODO debug code below
        #self.visualise_neural_net()

    def visualise_neural_net(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.DiGraph()
        unit = 1
        for node in self.input_nodes:
            node.layer = 1
            node.unit = unit
            G.add_node((1, unit), pos=(node.y, node.x))
            unit += 1
        layer = 1
        unit = 1
        last_y = -1
        for node in self.nodes:
            if last_y != node.y:
                layer += 1
                unit = 1
            node.layer = layer
            node.unit = unit
            #x = node.x if node.y != 1 else 0  # TODO this is hack to get nets with one output node looking pretty - rethink when multiple output nodes
            G.add_node((node.layer, node.unit), pos=(node.y, node.x))
            for link in node.ingoing_links:
                G.add_edge((link.out_node.layer, link.out_node.unit), (node.layer, node.unit), weight=link.weight)
            unit += 1
            last_y = node.y
        pos = nx.spring_layout(G, pos=dict(G.nodes(data='pos')), fixed=G.nodes)
        weights = np.array([G[u][v]['weight'] for u,v in G.edges]) * 4
        min_width = 0.1
        self.genome.visualise_genome(is_subplot=True)
        plt.subplot(2, 1, 2)
        plt.title('Neural Network Visualisation')
        nx.draw_networkx(G, pos=pos, node_size=650, node_color='#ffaaaa', linewidth=100, with_labels=True, width=min_width+weights)
        plt.show()

    def set_fitness(self, fitness):
        self.fitness = fitness-(len(self.links)*link_cost_coeff)  # fitness reward minus link/connection cost

    """
    def set_fitness(self, fitness):
        # Adjust fitness for number of species. NOTE no longer used as species no longer compete (local competition)
        self.fitness = self.fitness_unnorm/len(self.genome.species.genomes)
    """

    class Graph(nn.Module):
        """ computational graph """

        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = []  # torch tensor weights for each node
            self.activs = []  # torch activation funcs for each node
            self.outputs = torch.tensor((), dtype=torch.float32).new_empty((len(net.nodes) + net.n_net_inputs))
            self.output_inds = []  # Store node indices to get output of nodes going into this node
            # Setup torch tensors
            for node in net.nodes:
                # TODO!!!! we need to determine the activation function for each node from the genome
                node_weights = []
                in_node_inds = []
                for link in node.ingoing_links:
                    node_weights.append(link.weight)
                    in_node_inds.append(link.out_node.node_ind)
                    if link.out_node.node_ind is None:
                        print("")
                if len(in_node_inds) == 0 or in_node_inds is None:
                    print("")
                self.output_inds.append(torch.tensor(in_node_inds))
                self.weights.append(torch.tensor(node_weights, dtype=torch.float32))  # TODO requires_grad=True when adding gradient based lifetime learning
                self.activs.append(torch.relu)
            if net.discrete:
                from activations import step
                self.activs[-1] = step

        def forward(self, x):
            """ feedforward activation of graph and return output """
            n_inputs = len(x)
            # Update outputs vector with inputs
            self.outputs[torch.arange(n_inputs)] = torch.tensor(x, dtype=torch.float32)
            # loop each node and calculate output
            for i in range(len(self.activs)):
                y_unactiv = torch.dot(self.outputs[self.output_inds[i]], self.weights[i])
                y = self.activs[i](y_unactiv)
                self.outputs[i+n_inputs] = y
            return self.outputs[-self.net.n_net_outputs:]


class Link:
    """ Connection between two nodes """

    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight
        self.out_node = None  # out node
        self.in_node = None  # in node

    def __eq__(self, other):
        return True if self.x1 == other.x1 and self.x2 == other.x2 and self.y1 == other.y1 and self.y2 == other.y2 else False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """ optimised hashing for finding unique Links """
        return hash((self.x1, self.y1, self.x2, self.y2))


class Node:

    def __init__(self, x, y, act_func=torch.relu, node_ind=None):
        self.x = x
        self.y = y
        self.act_func = act_func
        self.ingoing_links = []  # links going into the node
        self.outgoing_links = []  # links going out of the node
        self.node_ind = node_ind  # node index, including input nodes
        self.layer = None  # Layer number. Only used in visualisation
        self.unit = None  # Position in layer. Only used in visualisation

    def __eq__(self, other):
        return True if self.x == other.x and self.y == other.y else False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """ optimised hashing for finding unique Nodes """
        return hash((self.x, self.y))

    def add_in_link(self, link):
        self.ingoing_links.append(link)

    def add_out_link(self, link):
        self.outgoing_links.append(link)

    def copy(self, link, is_in_node=True):
        """ Create a copy of the node without any links """
        node_copy = Node(self.x, self.y, act_func=self.act_func)
        if is_in_node:
            node_copy.add_in_link(link)
            link.in_node = node_copy
        else:
            node_copy.add_out_link(link)
            link.out_node = node_copy
        return node_copy

    def update_in_node(self, link):
        self.add_in_link(link)
        link.in_node = self

    def update_out_node(self, link):
        self.add_out_link(link)
        link.out_node = self

