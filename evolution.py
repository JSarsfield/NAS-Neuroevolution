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
import math
import random
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
        self.generation = 0
        self.pop_size = pop_size
        self.genomes = []  # Genomes in the current population
        self.neural_nets = []  # Neural networks (phenotype) in the current population
        self.species = []  # Group similar genomes into the same species
        self.n_net_inputs = n_net_inputs
        self.n_net_outputs = n_net_outputs
        self._get_initial_population()

    def _get_initial_population(self):
        while len(self.neural_nets) != self.pop_size:
            genome = CPPNGenome(self.gene_pool.gene_nodes_in, self.gene_pool.gene_nodes, self.gene_pool.gene_links)
            genome.create_initial_graph()
            net = Substrate().build_network_from_genome(genome, self.n_net_inputs, self.n_net_outputs)  # Express the genome to produce a neural network
            if not net.is_void:
                self.genomes.append(genome)
                self.neural_nets.append(net)

    def begin_evolution(self):
        print("Starting evolution...")
        while True:  # For infinite generations
            print("Start of generation ", str(self.generation))
            self._speciate_genomes()
            self._evaluate_population()
            print("End of generation ", str(self.generation))
            self._reproduce_new_generation()
            print("New generation reproduced")
            self.generation += 1

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
        """ reproduce next generation given fitnesses of current generation """
        new_genomes = []  # next gen genomes
        new_nets = []  # next gen nets
        new_species = []  # next gen speices
        self.neural_nets.sort(key=lambda net: net.fitness, reverse=True)  # Sort nets by fitness - element 0 = fittest
        print("Best fitness unnorm ", self.neural_nets[0].fitness_unnorm)
        # sort species genomes by fitness
        for s in self.species:
            s.genomes.sort(key=lambda x: x.net.fitness, reverse=True)  # Sort genomes within species by fitness
        # crossover
        i = 0
        stop = math.ceil(len(self.neural_nets)*pop_survival_thresh)
        while True:  # Until pop size reached # TODO Parallelise this just on the local machine e.g. python starmap
            # Crossover or self mutation
            if random.random() > interspecies_mating_prob:  # Mate within species
                if len(self.neural_nets[i].genome.species.genomes) != 1:  # For species with more than 1 genome
                    if self.neural_nets[i].genome.species.inds is None:
                        self.neural_nets[i].genome.species.inds = np.arange(0, math.ceil(len(self.neural_nets[i].genome.species.genomes)*pop_survival_thresh))
                    partner_ind = np.random.choice(self.neural_nets[i].genome.species.inds[self.neural_nets[i].genome.species.inds != i])  # ensure crossover with different genome in species
                    new_genome = self._crossover(self.neural_nets[i].genome, self.neural_nets[i].genome.species.genomes[partner_ind])
                else:  # Species only has 1 genome so copy and mutate
                    new_genome = self._copy_with_mutation(self.neural_nets[i].genome)
            else:  # Mate outside of species NOTE there is no guarantee the selected neural net is outside of species
                partner_ind = random.randint(0, stop)
                new_genome = self._crossover(self.neural_nets[i].genome, self.neural_nets[partner_ind])
            # Express the genome to produce a neural network
            new_net = Substrate().build_network_from_genome(new_genome, self.n_net_inputs, self.n_net_outputs)
            # Add new genome and net if not void
            if not new_net.is_void:
                new_genomes.append(new_genome)
                new_nets.append(new_net)
                i = 0 if i+1 == stop else i+1
                if len(new_genomes) == self.pop_size:
                    break
        # Overwrite current generation genomes/nets/species TODO pickle best performing
        self.genomes = new_genomes
        self.neural_nets = new_nets
        self.species = new_species

    def _crossover(self, g1, g2):
        """ crossover of two parent genomes """
        gene_nodes = set()
        gene_links = []
        i = 0
        j = 0
        max_genes = max(len(g1.gene_links), len(self.g2.gene_links))
        fittest = g1 if g1.net.fitness > g2.net.fitness else g2
        while i < len(g1.gene_links) and j < len(self.g2.gene_links):
            if g1.gene_links[i].historical_marker == g2.gene_links[j].historical_marker:

                i += 1
                j += 1
            elif g1.gene_links[i].historical_marker < g2.gene_links[j].historical_marker:
                i += 1
            else:
                j += 1
        # Mutate genome
        gene_nodes, gene_links = self._perform_mutations(gene_nodes, gene_links)
        return CPPNGenome(self.gene_pool.gene_nodes_in, gene_nodes, gene_links)

    def _copy_with_mutation(self, g1):
        """ copy a genome with mutation """
        # Mutate genome
        gene_nodes, gene_links = self._perform_mutations(gene_nodes, gene_links)
        return CPPNGenome(g1.gene_nodes_in, gene_nodes, gene_links)

    def _perform_mutations(self, gene_nodes, gene_links):
        """ perform structural and weight mutations """
        # mutate weights
        # mutate toggle links
        # mutate add links
        # mutate add nodes
        return gene_nodes, gene_links