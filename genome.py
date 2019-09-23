"""
CPPN implementation using PyTorch

CPPN is a biologically inspired genetic encoding/genome that produces neural network architectures when decoded.

See papers: 1. Compositional pattern producing networks: A novel abstraction of development by Kenneth O. Stanley
2. A hypercube-based encoding for evolving large-scale neural networks. Stanley, K., D’Ambrosio, D., & Gauci, J. (2009)

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
"""
import torch
import torch.nn as nn
"""
import copy  # deep copy genes
import operator # sort node genes by depth
from genes import GeneLink
import activations
from config import *
import functools

#torch.set_num_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


class CPPNGenome:
    """ CPPN genome - can express/decode to produce an ANN """

    def __init__(self, gene_nodes_in,
                 gene_nodes,
                 gene_links,
                 num_inputs=4,
                 num_outputs=2,
                 substrate_width=int(init_substrate_width_max/2),
                 substrate_height=int(init_substrate_height_max/2),
                 fitness=-9999):  #, var_thresh=0.3, band_thresh=0):
        self.weights = None  # Weight of links in graph. Sampled from parent/s genome/s or uniform distribution when no parent
        self.fitness = fitness  # Copy of fitness obtained from network
        self.gene_nodes = copy.deepcopy(gene_nodes)
        self.gene_nodes_in = copy.deepcopy(gene_nodes_in)
        self.gene_links = []
        self.substrate_width = substrate_width  # Number of horizontal rectangles on the substrate, CPPN decides if to express node
        self.substrate_height = substrate_height  # Number of vertical rectangles on the substrate
        #self.var_thresh = var_thresh  # ES-hyperneat parameter determining variance threshold for further splitting
        #self.band_thresh = band_thresh  # ES-hyperneat parameter
        self.species = None  # Species this genome belongs to
        self.act_set = activations.ActivationFunctionSet()
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
        #self.gene_links.sort(key=lambda x: x.historical_marker)  # Sorted genome required for speciation
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
        raise Exception("No node with historical marker found in func get_node_from_hist_marker genome.py hist maker: ", hist_marker)

    def create_initial_graph(self):
        """ Create an initial graph for generation zero that has no parent/s. Call on worker thread """
        # Initialise weights
        for link in self.gene_links:
            link.weight = random.uniform(weight_init_min, weight_init_max)
        # Initialise biases
        for node in self.gene_nodes:
            node.bias = random.uniform(bias_init_min, bias_init_max)
            if node.can_modify:
                node.act_func = self.act_set.get_random_activation_func()
        self.graph = Graph(self)
        #self.visualise_cppn()

    def create_graph(self):
        """ Create graph """
        self.graph = Graph(self)
        #self.visualise_cppn()

    def mutate_nonstructural(self):
        """ perform nonstructural mutations to existing gene nodes & links """
        # TODO consider clamping weights and biases?
        for link in self.gene_links:
            # Disable/Enable links
            if event(link_toggle_prob):  # Chance of toggling link
                link.enabled = True if link.enabled is False else False
            if link.enabled is False and event(link_enable_prob):  # Chance of enabling a disabled link
                link.enabled = True
            # Mutate weights
            if event(weight_mutate_rate):
                if event(weight_replace_rate):  # replace with random weight
                    link.weight = random.uniform(weight_init_min, weight_init_max)
                else:  # adjust weight
                    link.weight += random.uniform(-uniform_weight_scale, uniform_weight_scale)
        for node in self.gene_nodes:
            # Mutate bias
            if event(bias_mutate_rate):
                if event(bias_replace_rate):  # replace with random bias
                    node.bias = random.uniform(bias_init_min, bias_init_max)
                else:  # adjust bias
                    node.bias += random.uniform(-uniform_weight_scale, uniform_weight_scale)
            # Mutate activation func
            if node.can_modify:
                if event(change_act_prob):
                    node.act_func = self.act_set.get_random_activation_func()
                    # reinit freq amp and vshift when act func changes
                    if node.act_func.__name__[0] == "g":
                        node.freq = random.uniform(-gauss_freq_range, gauss_freq_range)
                        node.amp = random.uniform(-func_amp_range, func_amp_range)
                        node.vshift = random.uniform(-gauss_vshift_range, gauss_vshift_range)
                    elif node.act_func.__name__[0] == "s":
                        node.freq = random.uniform(-sin_freq_range, sin_freq_range)
                        node.amp = random.uniform(-func_amp_range, func_amp_range)
                        node.vshift = random.uniform(-sin_vshift_range, sin_vshift_range)
                # Adjust freq amp and vshift of activation function
                if event(func_adjust_prob):
                    if node.act_func.__name__[0] == "g":
                        node.freq += random.uniform(-guass_freq_adjust, guass_freq_adjust)
                    elif node.act_func.__name__[0] == "s":
                        node.freq += random.uniform(-sin_freq_adjust, sin_freq_adjust)
                if event(func_adjust_prob):
                    node.amp += random.uniform(-func_amp_adjust, func_amp_adjust)
                if event(func_adjust_prob):
                    node.vshift += random.uniform(-func_vshift_adjust, func_vshift_adjust)
        # Mutate substrate width/height rectangles
        if event(width_mutate_prob):
            if event(0.5):
                self.substrate_width += 1
            elif self.substrate_width > 1:
                self.substrate_width -= 1
        if event(height_mutate_prob):
            if event(0.5):
                self.substrate_height += 1
            elif self.substrate_height > 1:
                self.substrate_height -= 1
        """ ES-HyperNeat - no longer used
        # Mutate QuadTree variance
        if event(var_mutate_prob):
            self.var_thresh += np.random.normal(scale=gauss_var_scale)
            self.var_thresh = self.var_thresh if self.var_thresh > 0 else 0
        # Mutate QuadTree band thresh
        if event(band_mutate_prob):
            self.band_thresh += np.random.normal(scale=gauss_band_scale)
            self.band_thresh = self.band_thresh if self.band_thresh > 0 else 0
        """

    def set_species(self, species):
        """ set the species this genome belongs to """
        self.species = species

    def visualise_genome(self, is_subplot=False):
        """ Visualise genome graph """
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.DiGraph()
        unit = 1
        x_linspace = np.linspace(-1, 1, len(self.gene_nodes_in))
        labels = {}
        input_labs = ["x1", "y1", "x2", "y2"]
        for i, node in enumerate(self.gene_nodes_in):
            node.layer = 1
            node.unit = unit
            G.add_node((1, unit), pos=(node.depth, x_linspace[i]))
            labels[(1, unit)] = input_labs[i]
            unit += 1
        layer = 2
        unit = 1
        last_y = None
        x_spaces = []  # linspaces of X axis
        for node in self.gene_nodes:
            if last_y and last_y != node.depth:
                x_spaces.append(np.linspace(-1, 1, unit-1))
                layer += 1
                unit = 1
            node.layer = layer
            node.unit = unit
            unit += 1
            last_y = node.depth
        x_spaces.append(np.linspace(-1, 1, unit-1))
        for node in self.gene_nodes:
            G.add_node((node.layer, node.unit), pos=(node.depth, x_spaces[node.layer-2][node.unit-1]))
            labels[(node.layer, node.unit)] = node.act_func.__name__
            for link in node.ingoing_links:
                G.add_edge((link.out_node.layer, link.out_node.unit),
                           (node.layer, node.unit),
                           weight=link.weight,
                           color='r' if link.weight < 0 else 'b')
        pos = nx.spring_layout(G, pos=dict(G.nodes(data='pos')), fixed=G.nodes)
        weights = np.array([G[u][v]['weight'] for u, v in G.edges]) * 4
        plt.subplot(2, 1, 1)
        plt.title('Genome Graph Visualisation')
        colors = [G[u][v]['color'] for u, v in G.edges()]
        nx.draw_networkx(G,
                         pos=pos,
                         node_size=650,
                         node_color='#ffaaaa',
                         linewidth=100,
                         with_labels=True,
                         edge_color=colors,
                         width=weights,
                         labels=labels)
        if not is_subplot:
            plt.show()

    def visualise_cppn(self, resolution=(64, 64)):
        """ visualise the graph activations/link weights of a genome - see hyperneat paper"""
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import imshow
        data = np.empty([resolution[0], resolution[1]])
        x_linspace = np.linspace(-1, 1, resolution[0])
        y_linspace = np.linspace(-1, 1, resolution[1])
        for row, x in enumerate(x_linspace):
            for col, y in enumerate(y_linspace):
                data[row, col] = self.graph.forward([x, y, 0, 0])[0].item()
        #plt.axis([-1, 1, -1, 1])
        print(data.min(), " ", data.max())
        imshow(data, cmap='Greys', vmin=-1, vmax=1)
        plt.show()


class Graph(tf.keras.Model):
    """ computational graph TensorFlow """

    def __init__(self, genome):
        super(Graph, self).__init__()
        self.lyrs = []
        self.lyrs_meta = []
        # Create separate input layer for each input node
        for node in genome.gene_nodes_in:
            self.lyrs.append(layers.Input(shape=(1,)))
            self.lyrs_meta.append([node, self.lyrs[-1]])
            self.lyr_weights = []  # weights and bias
            # Create keras layers
        for node in genome.gene_nodes:
            self.lyr_weights.append([])
            concat = []
            for link in node.ingoing_links:
                self.lyr_weights[-1].append(link.weight)
                concat.append(list(filter(lambda x: x[0] is link.out_node, self.lyrs_meta))[0][-1])
            if node.act_func is any([activations.gaussian, activations.sin]):
                act_func = functools.partial(0, node.act_func, node.freq, node.amp, node.vshift)
            else:
                act_func = node.act_func
            if "diff" in node.node_func:
                self.lyrs.append(act_func(tf.reduce_sum([tf.multiply(concat[i], self.lyr_weights[i]) for i in range(len(self.lyr_weights))]))+node.bias)
            else:,1
                self.lyrs.append(layers.Dense(units=len(concat), activation=act_func, use_bias=False)(tf.stack(concat, axis=-1)))
            self.lyrs_meta.append([node, self.lyrs[-1]])
            # TODO set lyr weights and bias

    def build(self, input_shape):
        for i, l in enumerate(self.lyrs):
            l.build(tf.TensorShape(len(self.lyr_node_inds[i]), ))
            l.set_weights([np.transpose(self.lyr_weights[i]), l.get_weights()[1]])

    def call(self, inputs, training=False):
        pass

    """
    class Graph(nn.Module):
        # computational graph PyTorch

        def __init__(self, genome):
            super().__init__()
            self.genome = genome
            self.weights = []  # torch tensor weights for each node
            self.node_funcs = []  # torch node funcs
            self.activs = []  # torch activation funcs for each node
            self.activ_params = []  # freq, amp and vshift parameters for each node
            self.outputs = torch.tensor((), dtype=torch.float32).new_empty((len(genome.gene_nodes) + genome.cppn_inputs))
            self.output_inds = []  # Store node indices to get output of nodes going into this node
            self.node_biases = []
            # Setup torch tensors
            for node in genome.gene_nodes:
                node_weights = []
                in_node_inds = []
                all_links_disabled = True
                for link in node.ingoing_links:
                    if link.enabled:
                        node_weights.append(link.weight)
                        all_links_disabled = False
                    else:
                        node_weights.append(0)  # Disable link
                    in_node_inds.append(link.out_node.node_ind)
                if all_links_disabled:
                    self.node_biases.append(0)  # Disable node as all in going links are disabled
                else:
                    self.node_biases.append(node.bias)
                self.output_inds.append(torch.tensor(in_node_inds))
                self.weights.append(torch.tensor(node_weights, dtype=torch.float32))
                self.node_funcs.append(node.node_func)
                self.activs.append(node.act_func)
                if node.can_modify and node.act_func.__name__[0:2] == "ga" or node.act_func.__name__[0:2] == "si":
                    self.activ_params.append([node.freq, node.amp, node.vshift])
                else:
                    self.activ_params.append([])
                self.node_biases.append(node.bias)
            self.node_biases = torch.tensor(self.node_biases, dtype=torch.float32)

        def forward(self, x):
            # Query the CPPN
            n_inputs = len(x)
            # Update outputs vector with inputs
            self.outputs[torch.arange(n_inputs)] = torch.tensor(x, dtype=torch.float32)
            # loop each node and calculate output
            for i in range(len(self.activs)):
                y_unactiv = self.node_funcs[i](self.outputs[self.output_inds[i]], self.weights[i], self.node_biases[i])
                try:
                    y = self.activs[i](y_unactiv, *self.activ_params[i])
                except:
                    print("")
                self.outputs[i + n_inputs] = y
            return self.outputs[-self.genome.cppn_outputs:]
    """