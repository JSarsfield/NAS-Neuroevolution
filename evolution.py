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
from genes import GeneLink, GeneNode
from activations import ActivationFunctionSet, NodeFunctionSet


# TODO pickle top performing genomes after each/x generations
# TODO add connection cost to ensure
# TODO clamp weights to ensure minimum value

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
        self.act_set = ActivationFunctionSet()
        self.node_set = NodeFunctionSet()
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
            if not event(interspecies_mating_prob):  # Mate within species
                if len(self.neural_nets[i].genome.species.genomes) != 1:  # For species with more than 1 genome
                    if self.neural_nets[i].genome.species.inds is None:
                        self.neural_nets[i].genome.species.inds = np.arange(0, math.ceil(len(self.neural_nets[i].genome.species.genomes)*pop_survival_thresh))
                    if len(self.neural_nets[i].genome.species.inds) == 1:
                        self.neural_nets[i].genome.species.inds = np.append(self.neural_nets[i].genome.species.inds, 1)
                    partner_ind = np.random.choice(self.neural_nets[i].genome.species.inds[self.neural_nets[i].genome.species.inds != i])  # ensure crossover with different genome in species
                    new_genome = self._crossover(self.neural_nets[i].genome, self.neural_nets[i].genome.species.genomes[partner_ind])
                else:  # Species only has 1 genome so copy and mutate
                    new_genome = self._copy_with_mutation(self.neural_nets[i].genome)
            else:  # Mate outside of species NOTE there is no guarantee the selected neural net is outside of species
                partner_ind = random.randint(0, stop)
                new_genome = self._crossover(self.neural_nets[i].genome, self.neural_nets[partner_ind].genome)
            new_genome.create_graph()
            # Express the genome to produce a neural network
            new_net = Substrate().build_network_from_genome(new_genome, self.n_net_inputs, self.n_net_outputs)
            # Add new genome and net if not void
            if not new_net.is_void:
                new_genomes.append(new_genome)
                new_nets.append(new_net)
                i = 0 if i+1 == stop else i+1
                print("Added genome ",len(new_genomes), " of ", self.pop_size)
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
        nodes_to_add = []
        links_to_add = []
        while i < len(g1.gene_links) or j < len(g2.gene_links):
            if i < len(g1.gene_links) and j < len(g2.gene_links):
                if g1.gene_links[i].historical_marker == g2.gene_links[j].historical_marker:
                    if g1.net.fitness > g2.net.fitness:
                        links_to_add.append(g1.gene_links[i])
                        nodes_to_add.append(g1.gene_links[i].in_node)
                        nodes_to_add.append(g1.gene_links[i].out_node)
                    else:
                        links_to_add.append(g2.gene_links[j])
                        nodes_to_add.append(g1.gene_links[j].in_node)
                        nodes_to_add.append(g1.gene_links[j].out_node)
                    i += 1
                    j += 1
                elif g1.gene_links[i].historical_marker < g2.gene_links[j].historical_marker:
                    nodes_to_add.append(g1.gene_links[i].in_node)
                    nodes_to_add.append(g1.gene_links[i].out_node)
                    links_to_add.append(g1.gene_links[i])
                    i += 1
                else:
                    nodes_to_add.append(g2.gene_links[j].in_node)
                    nodes_to_add.append(g2.gene_links[j].out_node)
                    links_to_add.append(g2.gene_links[j])
                    j += 1
            else:
                if j == len(g2.gene_links):
                    nodes_to_add.append(g1.gene_links[i].in_node)
                    nodes_to_add.append(g1.gene_links[i].out_node)
                    links_to_add.append(g1.gene_links[i])
                    i += 1
                else:
                    nodes_to_add.append(g2.gene_links[j].in_node)
                    nodes_to_add.append(g2.gene_links[j].out_node)
                    links_to_add.append(g2.gene_links[j])
                    j += 1
        # Add in/out nodes and links
        for node in nodes_to_add:
            gene_nodes.add(GeneNode(node.depth,
                                    node.act_func,
                                    node.node_func,
                                    node.historical_marker,
                                    can_modify=node.can_modify,
                                    enabled=node.enabled,
                                    bias=node.bias))
        for link in links_to_add:
            gene_links.append((link.weight,
                               link.in_node.historical_marker,
                               link.out_node.historical_marker,
                               link.historical_marker,
                               link.enabled))
        gene_nodes = list(gene_nodes)
        gene_nodes.sort(key=lambda x: x.depth)
        self._mutate_structural(gene_nodes, gene_links)  # This is performed on master thread to ensure only new genes are added to gene pool
        gene_nodes_in = gene_nodes[:g1.cppn_inputs]
        gene_nodes = gene_nodes[g1.cppn_inputs:]
        new_genome = CPPNGenome(gene_nodes_in, gene_nodes, gene_links)
        new_genome.mutate_nonstructural()  # TODO this should be called on a worker thread
        return new_genome

    def _mutate_structural(self, gene_nodes, gene_links):
        """ mutate genome to add nodes and links. Note passed by reference """
        # TODO allow nodes to become disabled and thus all ingoing out going links disabled
        # TODO add config probs for toggling NODE! enable/disable
        # Mutate attempt add link
        if event(link_add_prob):
            # shuffle node indices
            inds = np.random.choice(len(gene_nodes), len(gene_nodes), replace=False)
            attempt = 0
            gene_links.sort(key=lambda x: x[1])
            for in_node_ind in inds:
                if gene_nodes[in_node_ind].depth == 0:
                    continue
                else:
                    ind_node_before = None
                    for i in range(in_node_ind, 0, -1):
                        if gene_nodes[i].depth != gene_nodes[in_node_ind].depth:
                            ind_node_before = i
                            break
                    # check if node has a potential new link
                    existing_links = []
                    for i, link in enumerate(gene_links):
                        if link[1] == gene_nodes[in_node_ind].historical_marker:
                            existing_links.append(link[2])
                    # If number of existing ingoing links is less than the number of nodes before this node then add a new link
                    if len(existing_links) != ind_node_before+1:
                        existing_links = [i for i, node in enumerate(gene_nodes) if node.historical_marker in existing_links]
                        out_node = gene_nodes[np.random.choice(np.setdiff1d(np.arange(ind_node_before+1), existing_links), 1)[0]]
                        # Add new link
                        new_link = self.gene_pool.get_or_create_gene_link(gene_nodes[in_node_ind].historical_marker, out_node.historical_marker)
                        gene_links.append((random.uniform(weight_init_min, weight_init_max),
                                           new_link.in_node.historical_marker,
                                           new_link.out_node.historical_marker,
                                           new_link.historical_marker,
                                           new_link.enabled))
                        break
                    attempt += 1
                if attempt == new_link_attempts:
                    break
        # Mutate add node with random activation function
        if event(node_add_prob):
            # Get a random link to split and add node
            gene_link_ind = random.randint(0, len(gene_links)-1)
            old_link = gene_links[gene_link_ind]
            in_node = self.gene_pool.get_node_from_hist_marker(old_link[1])
            out_node = self.gene_pool.get_node_from_hist_marker(old_link[2])
            old_link_update = (old_link[0], old_link[1], old_link[2], old_link[3], False)
            del gene_links[gene_link_ind]
            gene_links.append(old_link_update)
            # Create new node
            gene_nodes.append(self.gene_pool.create_gene_node({"depth": out_node.depth+((in_node.depth-out_node.depth)/2),
                                             "activation_func": self.act_set.get_random_activation_func(),
                                             "node_func": self.node_set.get("dot"),
                                             "bias": random.uniform(bias_init_min, bias_init_max)}))
            # Create new link going into new node
            new_link = self.gene_pool.create_gene_link({"weight": 1,
                                   "in_node": in_node,
                                   "out_node": gene_nodes[-1]})
            gene_links.append((new_link.weight,
                               new_link.in_node.historical_marker,
                               new_link.out_node.historical_marker,
                               new_link.historical_marker,
                               new_link.enabled))
            # Create new link going out of new node
            new_link = self.gene_pool.create_gene_link({"weight": old_link[0],
                                                        "in_node": gene_nodes[-1],
                                                        "out_node":  out_node})
            gene_links.append((new_link.weight,
                               new_link.in_node.historical_marker,
                               new_link.out_node.historical_marker,
                               new_link.historical_marker,
                               new_link.enabled))
        gene_nodes.sort(key=lambda x: x.depth)

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