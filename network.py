""" Create Artifcial Neural Network

Artificial Neural Network / Phenotype. Expressed given a genome.

 NOTE self.fitness calc is slightly different from original NEAT - see adjusted fitness section 3.3 pg 12 "Evolving Neural Networks through Augmenting Topologies" 2002

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import tensorflow as tf
import random
import numpy as np
from config import link_cost_coeff


def relu(x):
    return max(0, x)


def step_zero(x):
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
        self.graph = None

    def init_graph(self):
        """ We need to arrange the network inputs and weights for each layer into balanced matrices for calculating
        layer outputs efficiently with TensorFlow
        """
        layer_sizes = []  # Number of nodes in layer
        layer_link_size = []  # Determined by node in layer with most ingoing links, for efficient TF layer calculations
        layer_info = []  # ingoing link weights for each node in each layer as a vector
        lyr_node_inds = []  # node_inds for layer, used to gather indices for input into layer
        most_links = 0
        unit = 0
        for node in self.input_nodes:
            node.layer = 0
            node.unit = unit
            unit += 1
        layer = 0
        unit = 0
        last_y = -1
        for node in self.nodes:
            if last_y != node.y:
                if last_y != -1:
                    lyr_node_inds[-1] = list(lyr_node_inds[-1])
                    layer_sizes.append(unit)
                    layer_link_size.append(most_links)
                layer_info.append({})  # Add layer
                lyr_node_inds.append(set())  # unique node_inds in the layer
                layer_info[-1]["node_inds"] = []  # 2D array whereby each row is a node in the layer
                layer_info[-1]["weights"] = []  # 2D array whereby each row is a node in the layer
                layer += 1
                unit = 0
            if len(node.ingoing_links) > most_links:
                most_links = len(node.ingoing_links)
            layer_info[-1]["node_inds"].append([]) # Node info in layer
            layer_info[-1]["weights"].append([])  # Node info in layer
            for link in node.ingoing_links:
                lyr_node_inds[-1].add(link.out_node.node_ind)
                layer_info[-1]["node_inds"][-1].append(link.out_node.node_ind)  # Add in node ind and link weight to node in layer
                layer_info[-1]["weights"][-1].append(link.weight)
            node.layer = layer
            node.unit = unit
            unit += 1
            last_y = node.y
        if last_y != -1:
            lyr_node_inds[-1] = list(lyr_node_inds[-1])
            layer_sizes.append(unit)
            layer_link_size.append(most_links)
        lyr_weights = []
        for i, lyr in enumerate(layer_info):
            lyr_weights.append(np.empty([len(lyr["node_inds"]), len(lyr_node_inds[i])], dtype=np.float32))
            for j, node in enumerate(lyr["node_inds"]):
                new_inds = [in_node for in_node in lyr_node_inds[i] if in_node not in node]
                node.extend(new_inds)
                lyr["weights"][j].extend([0]*len(new_inds))
                np_node = np.array(node)
                lyr["weights"][j] = np.array(lyr["weights"][j], dtype=np.float32)
                lyr_weights[-1][j] = np.copy(lyr["weights"][j][np.argsort(np_node)])
        self.graph = Graph(self.n_net_inputs, layer_sizes, lyr_node_inds, lyr_weights)

    def visualise_neural_net(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.DiGraph()
        for node in self.input_nodes:
            G.add_node((node.layer, node.unit), pos=(node.y, node.x))
        for node in self.nodes:
            G.add_node((node.layer, node.unit), pos=(node.y, node.x))
            for link in node.ingoing_links:
                G.add_edge((link.out_node.layer, link.out_node.unit),
                           (node.layer, node.unit),
                           weight=link.weight,
                           color='r' if link.weight < 0 else 'b')
        pos = nx.spring_layout(G, pos=dict(G.nodes(data='pos')), fixed=G.nodes)
        weights = np.array([G[u][v]['weight'] for u,v in G.edges]) * 4
        self.genome.visualise_genome(is_subplot=True)
        plt.subplot(2, 1, 2)
        plt.title('Neural Network Visualisation')
        colors = [G[u][v]['color'] for u, v in G.edges()]
        nx.draw_networkx(G,
                         pos=pos,
                         node_size=650,
                         node_color='#ffaaaa',
                         linewidth=100,
                         with_labels=True,
                         edge_color=colors,
                         width=weights)
        plt.show()

    def set_fitness(self, fitness):
        self.fitness = fitness-(len(self.links)*link_cost_coeff)  # fitness reward minus link/connection cost
        self.genome.fitness = self.fitness

    """
    def set_fitness(self, fitness):
        # Adjust fitness for number of species. NOTE no longer used as species no longer compete (local competition)
        self.fitness = self.fitness_unnorm/len(self.genome.species.genomes)
    """


class Graph(tf.keras.Model):
    """ computational graph of neural network """

    def __init__(self, n_net_inputs, layer_sizes, lyr_node_inds, lyr_weights):
        """
        layer_sizes = list of num of nodes in each layer (excl. input layer)
        lyr_node_inds = list of max num of ingoing links for a node within the layer (excl. input layer)
        lyr_weights = list of each layer's initialisation values for weights and biases Note bias currently unused
        """
        super(Graph, self).__init__()
        self.n_net_inputs = n_net_inputs
        self.layer_sizes = layer_sizes
        self.lyr_node_inds = [tf.constant(l, dtype=tf.int32) for l in lyr_node_inds]
        self.lyr_weights = lyr_weights
        self.lyrs = []
        self.outputs = None
        self.out_update_inds = [tf.constant(tf.expand_dims(tf.range(0, n_net_inputs, dtype=tf.int32), axis=1))]
        start_ind = n_net_inputs
        for i, size in enumerate(layer_sizes[:-1]):
            finish_ind = start_ind+size
            self.out_update_inds.append(tf.constant(tf.expand_dims(tf.range(start_ind, finish_ind, dtype=tf.int32), axis=1)))
            start_ind = finish_ind
        for size in layer_sizes:
            self.lyrs.append(tf.keras.layers.Dense(units=size,
                                                   activation=tf.nn.tanh,
                                                   use_bias=True,
                                                   kernel_initializer='zeros',
                                                   bias_initializer='zeros',
                                                   input_shape=(n_net_inputs,)))

    def build(self, input_shape):
        # init layer weights and biases
        for i, l in enumerate(self.lyrs):
            l.build(tf.TensorShape(len(self.lyr_node_inds[i]),))
            l.set_weights([np.transpose(self.lyr_weights[i]), l.get_weights()[1]])
        self.outputs = tf.Variable(initial_value=tf.zeros([self.n_net_inputs+sum(self.layer_sizes)]),
                                   trainable=False,
                                   validate_shape=True,
                                   name="flattened_outputs_vector")

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)
        # Call hidden layers
        for i, l in enumerate(self.lyrs[:-1]):
            tf.compat.v1.scatter_nd_update(self.outputs, self.out_update_inds[i], x)
            x = tf.gather(self.outputs, self.lyr_node_inds[i])
            x = l(tf.expand_dims(x, axis=0))[-1]
        tf.compat.v1.scatter_nd_update(self.outputs, self.out_update_inds[-1], x)
        x = tf.gather(self.outputs, self.lyr_node_inds[-1])
        return self.lyrs[-1](tf.expand_dims(x, axis=0))[-1]  # call output layer and return result

        """
        def __init__(self, net):
            super().__init__()
            self.layers = []
            current_layer_depth = net.nodes[0].y
            nodes_in_layer = 0
            # Add hidden layers
            for node in net.nodes:
                if current_layer_depth == node.y:
                    nodes_in_layer += 1
                else:
                    self.layers.append(nn.Tanh(nodes_in_layer))
                    nodes_in_layer = 1
                    current_layer_depth = node.y
            # Add output layer
            self.layers.append(nn.Tanh(nodes_in_layer))
            print("")


        def forward(self, x):
            return x
        """

    """
    class GraphOld(nn.Module):
        

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
                self.activs.append(node.act_func)
            if net.discrete:
                from activations import step
                self.activs[-1] = step

        def forward(self, x):
            
            n_inputs = len(x)
            # Update outputs vector with inputs
            self.outputs[torch.arange(n_inputs)] = torch.tensor(x, dtype=torch.float32)
            # loop each node and calculate output
            for i in range(len(self.activs)):
                y_unactiv = torch.dot(self.outputs[self.output_inds[i]], self.weights[i])
                y = self.activs[i](y_unactiv)
                self.outputs[i+n_inputs] = y
            return self.outputs[-self.net.n_net_outputs:]
    """


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

    def __init__(self, x, y, act_func=step_zero, node_ind=None):
        self.x = x
        self.y = y
        self.act_func = step_zero  # act_func if y != 1 else torch.tanh
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

