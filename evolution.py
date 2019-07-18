"""
Control evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
import tensorflow as tf
import pickle
from genes import GenePool
from genome import CPPNGenome
#from time import perf_counter  # Accurate timing
from substrate import Substrate


class Evolution:

    # TODO create population of CPPN tensorflow graphs with mutation
    # TODO create population of ANN tensorflow graphs given the CPPNs for evaluation

    def __init__(self, n_net_inputs, n_net_outputs, pop_size=10, dataset=None, yaml_config=None):
        self.gene_pool = GenePool(num_inputs=4)  # inputs x1 x2 y1 y2
        self.generation = -1
        self.pop_size = pop_size
        self.genomes = []  # Genomes in the current population
        self.neural_nets = []  # Neural networks (phenotype) in the current population
        self.n_net_inputs = n_net_inputs
        self.n_net_outputs = n_net_outputs
        self._get_initial_population()

    def _get_initial_population(self):
        for i in range(self.pop_size):
            self.genomes.append(CPPNGenome(self.gene_pool.geneNodesIn, self.gene_pool.geneNodes, self.gene_pool.geneLinks))
            self.genomes[-1].create_initial_graph()
            self.neural_nets.append(Substrate().build_network_from_genome(self.genomes[-1], self.n_net_inputs, self.n_net_outputs))  # Express the genome to produce a neural network

            """
            #input = tf.Variable(np.random.uniform(1,-1, 4), dtype=tf.float32, shape=(1,self.genomes[-1].num_inputs), name="input")  # np.expand_dims(np.random.uniform(1,-1, 4).astype(np.float32), axis=0)
            input = np.array([0,0,1,1])
            res = self.genomes[-1].graph.query(input)
            print(res)
            """

    def begin_evolution(self):
        while True: # For infinite generations
            self.generation += 1
            self._reproduce_new_generation()

    def _reproduce_new_generation(self):
        for i in range(self.pop_size):  # TODO Parallelise this just on the local machine e.g. python starmap
            pass