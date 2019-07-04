"""
TensorFlow Keras model implementation of CPPN

CPPN is a biologically inspired genetic encoding/genome that produces neural network architectures when decoded.

See paper: Compositional pattern producing networks: A novel abstraction of development by Kenneth O. Stanley

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import tensorflow as tf


class CPPNGenome:
    """ CPPN genome that can be expressed/decoded to produce an ANN """

    def __init__(self, geneNodesInOut, geneLinks, geneNodes, num_inputs=3, num_outputs=1):
        """ Call on master thread then call a create graph function on the worker thread """
        self.weights = None  # Weight of links in graph. Sampled from parent/s genome/s or uniform distribution when no parent
        self.geneNodesInOut = geneNodesInOut
        self.geneLinks = geneLinks
        self.geneNodes = geneNodes
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph = None  # Store TensorFlow graph. Created on worker thread within a create graph function

    def create_initial_graph(self, geneLinks):
        """ Create an initial graph for generation zero that has no parent/s. Call on worker thread """
        self.graph = tf.Graph()  # TensorFlow computational graph for querying the CPPN. tf.Graph NOT THREAD SAFE, CREATE THE GRAPH ON THE WORKER THREAD
        with self.graph.as_default():
            x = tf.place

        self.weights = tf.random.uniform((-1, 1))
        self.graph.finalize()

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
        """ Optimised TensorFlow computational graph representing the CPPN """
        def __init__(self):  # Pass variables outside of the graph here
            pass

        @tf.function
        def __call__(self, x1, y1, x2, y2): # TODO tf.Tensor as input
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
