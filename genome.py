"""
TensorFlow Keras model implementation of CPPN

CPPN is a biologically inspired genetic encoding/genome that produces neural network architectures when decoded.

See paper: Compositional pattern producing networks: A novel abstraction of development by Kenneth O. Stanley

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
import tensorflow as tf
import copy  # deep copy genes
import operator # sort node genes by depth
import random  # random uniform weight


class CPPNGenome:
    """ CPPN genome that can be expressed/decoded to produce an ANN """

    def __init__(self, geneNodesIn, geneNodes, geneLinks, num_inputs=3, num_outputs=1):
        """ Call on master thread then call a create graph function on the worker thread """
        self.weights = None  # Weight of links in graph. Sampled from parent/s genome/s or uniform distribution when no parent
        self.geneNodesIn = copy.deepcopy(geneNodesIn)
        self.geneLinks = copy.deepcopy(geneLinks)
        self.geneNodes = copy.deepcopy(geneNodes)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph = None  # Store TensorFlow graph. Created on worker thread within a create graph function

    def create_initial_graph(self):
        """ Create an initial graph for generation zero that has no parent/s. Call on worker thread """
        # TODO mutate first, structural mutation?
        for link in self.geneLinks:
            link.weight = random.uniform(-1, 1)
        self.geneNodes.sort(key=operator.attrgetter('depth'))
        self.graph = CPPNGenome.Graph(self)
        """
        self.graph = tf.Graph()  # TensorFlow computational graph for querying the CPPN. tf.Graph NOT THREAD SAFE, CREATE THE GRAPH ON THE WORKER THREAD
        with self.graph.as_default():
            x = tf.place

        self.weights = tf.random.uniform((-1, 1))
        self.graph.finalize()
        """

    def _create_graph(self, parent_genome):
        """ Create new graph given single parent genome. Call on worker thread """
        pass  # self.genes

    def _create_graph_from_parents(self, parent_genome1, parent_genome2):
        """ Create new graph given two parent genomes.  Call on worker thread """
        pass  # self.genes

    def _perturb_weights(self):
        """ Modify the weights of the links within the CPPN """
        pass

    class Graph:
        """ Optimised TensorFlow computational graph representing the CPPN. tf.Graph NOT THREAD SAFE, CREATE THE GRAPH ON THE WORKER THREAD """
        def __init__(self, genome):  # Pass variables outside of the graph here
            self.genome = genome
            self.weights = []
            self.nodes = []
            self.link_map = []
            # TODO get num of layers and num of layer with most nodes. For creating TensorFlow activation matrix
            self.graph_cols = genome.num_inputs  # layer with most nodes
            self.graph_rows = 1  # number of layers
            last_row_depth = 0
            nodes_in_row = 1
            # TODO create link map which has the indices of links going into each node
            # Setup tensorflow constants e.g. activation funcs & weights (CPPN weights only change during crossover)
            for node in genome.geneNodes:
                if node.depth != last_row_depth:
                    self.graph_rows += 1
                    last_row_depth = node.depth
                else:
                    nodes_in_row += 1
                    if nodes_in_row > self.graph_cols:
                        self.graph_cols = nodes_in_row
                self.weights.append(tf.constant(np.fromiter((link.weight for link in node.ingoing_links), np.float32, len(node.ingoing_links)), tf.float32))
                self.nodes.append(node.activation_func)
            print("")

        #@tf.function
        def query(self, input):  # input is a Tensor with x1, x2, y1, y2
            """ Query the CPPN """
            input = tf.Variable(input, dtype=tf.float32, name="input")
            activs = tf.zeros((self.graph_cols, self.graph_rows), dtype=tf.float32, name="activs")
            for ind_node, node in enumerate(self.genome.geneNodes):
                for ind_link, link in enumerate(node.ingoing_links):
                    print("")
            return

    """
    @tf.function
    def simple_nn_layer(x, y):
        return tf.nn.sigmoid(tf.matmul(x, y))

    
    def __init__(self, genes):
        self.model = tf.keras.Sequential((
            tf.keras.layers.Dense(1, input_shape=(1, 4), activation=tf.nn.sigmoid))) #  kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1), bias_initializer=tf.keras.initializers.RandomUniform(-1, 1))
        self.model.build()
        
        self.model = tf.keras.Sequential()

        # Add output layer which must be sigmoid to squash between 0 and 1
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1), bias_initializer=tf.keras.initializers.RandomUniform(-1, 1)))
        self.model.build(input_shape=(1,4))
    """
