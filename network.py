""" Create Artifcial Neural Network

Artificial Neural Network / Phenotype. Expressed given a genome.

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np

class Network:
    # TODO determine how activation func of nodes is going to be determined
    # TODO if number of outputs is greater than 1 is there always going to be a path to all output nodes??
    # TODO GRADIENT BASED LIFETIME LEARNING - requires bias input with constant value 1 for each hidden node, give it random trainable weight. Don't put bias nodes on output layer.

    def __init__(self, genome, links, nodes, n_net_inputs, n_net_outputs, void=False):
        self.is_void = void
        if void:
            self.score = -1
            return
        self.genome = genome  # Genome used to express ANN
        self.links = links
        self.nodes = nodes
        for i, node in enumerate(self.nodes):
            node.node_ind = i
        self.input_nodes = self.nodes[:n_net_inputs]
        del self.nodes[:n_net_inputs] # Remove input nodes
        self.n_net_inputs = n_net_inputs
        self.n_net_outputs = n_net_outputs
        self.score = None  # Score the network after evaluating during lifetime
        self.graph = Network.Graph(self)

        self.visualise_neural_net()
        self.graph.forward([1,2,3,4])


    def create_graph(self):
        pass

    def visualise_neural_net(self):
        G = nx.DiGraph()
        #for node in self.nodes:
        #    G.add_node(node.node_ind)
        unit = 1
        for node in self.input_nodes:
            node.layer = 1
            node.unit = unit
            G.add_node((1, unit), pos=(node.y, node.x))
            unit += 1
        layer = 2
        unit = 1
        last_y = None
        for node in self.nodes:
            if last_y and last_y != node.y:
                layer += 1
                unit = 1
            node.layer = layer
            node.unit = unit
            G.add_node((node.layer, node.unit), pos=(node.y, node.x))
            for link in node.ingoing_links:
                G.add_edge((link.outgoing_node.layer, link.outgoing_node.unit), (node.layer, node.unit), weight=link.weight)
            unit += 1
            last_y = node.y
        pos = nx.spring_layout(G, pos=dict(G.nodes(data='pos')), fixed=G.nodes)
        weights = np.array([G[u][v]['weight'] for u,v in G.edges]) * 4
        nx.draw(G, pos=pos, node_size=650, node_color='#ffaaaa', linewidth=100, with_labels=True, width=weights)
        plt.show()

    class Graph(nn.Module):
        """ computational graph """

        def __init__(self, net):
            super().__init__()
            self.net = net
            self.weights = []  # torch tensor weights for each node
            self.activs = []  # torch activation funcs for each node
            self.outputs = torch.tensor((), dtype=torch.float32).new_empty((len(net.nodes) + net.n_net_inputs))
            self.output_inds = []  # Store node indices to get output of nodes going into this node
            node_ind = net.n_net_inputs  # node index counter, the first indices are reserved for input values
            # Setup torch tensors
            for node in net.nodes:
                # TODO!!!! we need to determine the activation function for each node from the genome
                node_weights = []
                ingoing_node_inds = []
                for link in node.ingoing_links:
                    node_weights.append(link.weight)
                    ingoing_node_inds.append(link.outgoing_node.node_ind)
                    node_ind += 1
                self.output_inds.append(torch.tensor(ingoing_node_inds))
                self.weights.append(torch.tensor(node_weights, dtype=torch.float32))  # TODO requires_grad=True when adding gradient based lifetime learning
                self.activs.append(torch.sigmoid)

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
        self.outgoing_node = None  # outgoing node
        self.ingoing_node = None  # ingoing node


class Node:

    def __init__(self, x, y, act_func=F.sigmoid):
        self.x = x
        self.y = y
        self.layer = None  # Layer number
        self.unit = None  # Position in layer
        self.act_func = act_func
        self.ingoing_links = []  # links going into the node
        self.outgoing_links = []  # links going out of the node
        self.node_ind = None  # node index, including input nodes
