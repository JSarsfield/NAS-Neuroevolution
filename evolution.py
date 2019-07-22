"""
Control evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
import pickle
from genes import GenePool
from genome import CPPNGenome
from collections import deque
#from time import perf_counter  # Accurate timing
from substrate import Substrate
from environment import EnvironmentReinforcement
from species import Species
from config import *


# TODO pickle top performing genomes after each/x generations
# TODO add connection cost to ensure

class Evolution:

    def __init__(self, n_net_inputs, n_net_outputs, pop_size=10, dataset=None, yaml_config=None):
        self.gene_pool = GenePool(cppn_inputs=4)  # CPPN inputs x1 x2 y1 y2
        self.generation = -1
        self.pop_size = pop_size
        self.genomes = []  # Genomes in the current population
        self.neural_nets = []  # Neural networks (phenotype) in the current population
        self.species = []  # Group similar genomes into the same species
        self.n_net_inputs = n_net_inputs
        self.n_net_outputs = n_net_outputs
        self._get_initial_population()

    def _get_initial_population(self):
        while len(self.neural_nets) != self.pop_size:
            genome = CPPNGenome(self.gene_pool.gene_nodes_in, self.gene_pool.gene_nodes, self.gene_pool.gene_links, num_inputs=4, num_outputs=2)
            genome.create_initial_graph()
            net = Substrate().build_network_from_genome(genome, self.n_net_inputs, self.n_net_outputs)  # Express the genome to produce a neural network
            if not net.is_void:
                self.genomes.append(genome)
                self.neural_nets.append(net)

    def begin_evolution(self):
        print("Starting evolution...")
        while True: # For infinite generations
            print("Generation ", str(self.generation))
            self._speciate_genomes()
            self._evaluate_population()
            self.generation += 1
            self._reproduce_new_generation()

    def _speciate_genomes(self):
        """ Put genomes into species """
        genomes_unmatched = deque(self.genomes)
        # Put all unmatched genomes into a species or create new species if no match
        while genomes_unmatched:
            genome = genomes_unmatched.pop()
            matched = False
            # Search existing species to find match for this genome
            for s in self.species:
                if s.get_distance(genome) < compatibility_thresh:
                    s.add_to_species(genome)
                    matched = True
                    break
            # No species found so create new species and use this genome as the representative genome
            if not matched:
                self.species.append(Species(genome))

    def _evaluate_population(self):
        """ evaluate all neural networks in population and store fitnesses """
        for net in self.neural_nets:
            env = EnvironmentReinforcement()
            env.evaluate(net)

    def _reproduce_new_generation(self):
        self.neural_nets.sort(key=lambda net: net.fitness, reverse=True)  # Sort nets by fitness - element 0 = fittest
        for i in range(self.pop_size):  # TODO Parallelise this just on the local machine e.g. python starmap
            # Rank nets/genomes by net fitness
            pass
