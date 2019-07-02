"""
Control evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
import tensorflow as tf
from genes import GenePool
import cppn


class Evolution:

    # TODO create population of CPPN tensorflow graphs with mutation
    # TODO create population of ANN tensorflow graphs given the CPPNs for evaluation

    def __init__(self, num_inputs, pop_size=10, dataset=None, yaml_config=None):
        self.gene_pool = GenePool(num_inputs=4)  # inputs x1 x2 y1 y2
        self.generation = -1
        self.pop_size = pop_size

    def begin_evolution(self):
        while True: # For infinite generations
            self.generation += 1
            self._get_new_generation()

    def _get_new_generation(self):
        