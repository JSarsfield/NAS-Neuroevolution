"""
TensorFlow Keras model implementation of CPPN

CPPN is a biologically inspired genetic encoding that produces neural network architectures when decoded.

See paper: Compositional pattern producing networks: A novel abstraction of development by Kenneth O. Stanley

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import tensorflow as tf


class CPPN:

    def __init__(self, _genes):
        self.genes = _genes
        self.create_graph()

    def create_graph(self):
        self.weights = tf.random.uniform((-1, 1))

    @tf.function
    def call_cppn(self, x1, y1, x2, y2):
        return

    @tf.function
    def simple_nn_layer(x, y):
        return tf.nn.sigmoid(tf.matmul(x, y))

    """
    def __init__(self, genes):
        self.model = tf.keras.Sequential((
            tf.keras.layers.Dense(1, input_shape=(1, 4), activation=tf.nn.sigmoid))) #  kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1), bias_initializer=tf.keras.initializers.RandomUniform(-1, 1))
        self.model.build()
        
        self.model = tf.keras.Sequential()

        # Add output layer which must be sigmoid to squash between 0 and 1
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.RandomUniform(-1, 1), bias_initializer=tf.keras.initializers.RandomUniform(-1, 1)))
        self.model.build(input_shape=(1,4))
    """
