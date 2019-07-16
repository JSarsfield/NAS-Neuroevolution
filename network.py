""" Create Artifcial Neural Network

Artificial Neural Network / Phenotype. Expressed given a genome.

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
import tensorflow as tf
import random


class Network:
    # TODO determine how the neural network connectome is going to be represented
    # TODO determine how activation func of nodes is going to be determined

    def __init__(self, genome, links):
        self.genome = genome  # Genome used to express ANN
        self.links = links

    def create_graph(self):
        pass


class Link:
    """ Connection between two nodes """

    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight
        self.outgoing_node = None  # index of outgoing node
        self.ingoing_node = None  # index of ingoing node


class Node:

    def __init__(self, x, y, act_func=None):
        self.x = x
        self.y = y
        self.layer = None  # Layer number
        self.act_func = act_func
        self.ingoing_links = []  # links going into the node
        self.outgoing_links = []  # links going out of the node