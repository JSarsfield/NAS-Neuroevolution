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
from environment import EnvironmentReinforcement, get_env_spaces, parallel_evaluate_net
from species import Species
from config import *
from genes import GeneLink, GeneNode
from activations import ActivationFunctionSet, NodeFunctionSet
import keyboard


# TODO pickle top performing genomes after each/x generations
# TODO !!!! add connection cost to ensure
# TODO clamp weights to ensure minimum value
# TODO investigate changing and dynamic environments
# TODO select for novelty/diversity

def parallel_reproduce_eval(parent_genomes, n_net_inputs, n_net_outputs, env, gym_env_string):
    # Reproduce from parent genomes
    if type(parent_genomes) is tuple:  # Two parent genomes so crossover
        genome = crossover(parent_genomes[0], parent_genomes[1])
    else:  # One parent genome so mutate
        genome = copy_with_mutation(parent_genomes)
    # Create net from genome
    net = Substrate().build_network_from_genome(genome, n_net_inputs, n_net_outputs)
    # Evaluate
    fitness = env(gym_env_string).evaluate(net)
    net.set_fitness(fitness)
    return genome, net


def crossover(g1, g2):
    """ crossover of two parent genomes """
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
                    nodes_to_add.append(g2.gene_links[j].in_node)
                    nodes_to_add.append(g2.gene_links[j].out_node)
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
    return create_new_genome(nodes_to_add, links_to_add, g1.cppn_inputs)


def copy_with_mutation(g1):
    """ copy a genome with mutation """
    return create_new_genome(g1.gene_nodes_in + g1.gene_nodes, g1.gene_links, g1.cppn_inputs)


def create_new_genome(nodes_to_add, links_to_add, cppn_inputs):
    """ Create new genome - perform structural & non structural mutation """
    gene_nodes = set()
    gene_links = []
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
    mutate_structural(gene_nodes, gene_links)  # This is performed on master thread to ensure only new genes are added to gene pool
    gene_nodes_in = gene_nodes[:cppn_inputs]
    gene_nodes = gene_nodes[cppn_inputs:]
    new_genome = CPPNGenome(gene_nodes_in, gene_nodes, gene_links)
    new_genome.mutate_nonstructural()  # TODO this should be called on a worker thread
    return new_genome


def mutate_structural(gene_nodes, gene_links):
    """ mutate genome to add nodes and links. Note passed by reference """
    # TODO !!! allow nodes to become disabled and thus all ingoing out going links disabled
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
                existing_links = np.unique(existing_links)
                # If number of existing ingoing links is less than the number of nodes before this node then add a new link
                if len(existing_links) != ind_node_before+1:
                    existing_links = [i for i, node in enumerate(gene_nodes) if node.historical_marker in existing_links]
                    out_node = gene_nodes[np.random.choice(np.setdiff1d(np.arange(ind_node_before+1), existing_links), 1)[0]]
                    # Add new link
                    new_link = gene_pool.get_or_create_gene_link(gene_nodes[in_node_ind].historical_marker, out_node.historical_marker)
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
        in_node = gene_pool.get_node_from_hist_marker(old_link[1])
        out_node = gene_pool.get_node_from_hist_marker(old_link[2])
        old_link_update = (old_link[0], old_link[1], old_link[2], old_link[3], False)
        del gene_links[gene_link_ind]
        gene_links.append(old_link_update)
        # Create new node
        gene_nodes.append(gene_pool.create_gene_node({"depth": out_node.depth+((in_node.depth-out_node.depth)/2),
                                         "activation_func": act_set.get_random_activation_func(),
                                         "node_func": node_set.get("dot"),
                                         "bias": random.uniform(bias_init_min, bias_init_max)}))
        # Create new link going into new node
        new_link = gene_pool.create_gene_link({"weight": 1,
                               "in_node": in_node,
                               "out_node": gene_nodes[-1]})
        gene_links.append((new_link.weight,
                           new_link.in_node.historical_marker,
                           new_link.out_node.historical_marker,
                           new_link.historical_marker,
                           new_link.enabled))
        # Create new link going out of new node
        new_link = gene_pool.create_gene_link({"weight": old_link[0],
                                                    "in_node": gene_nodes[-1],
                                                    "out_node":  out_node})
        gene_links.append((new_link.weight,
                           new_link.in_node.historical_marker,
                           new_link.out_node.historical_marker,
                           new_link.historical_marker,
                           new_link.enabled))
    gene_nodes.sort(key=lambda x: x.depth)


class Evolution:

    def __init__(self, n_net_inputs, n_net_outputs, pop_size=10, environment=None, gym_env_string="BipedalWalker-v2", dataset=None, yaml_config=None, parallel=True, processes=4):
        self.gene_pool = GenePool(cppn_inputs=4)  # CPPN inputs x1 x2 y1 y2
        self.generation = 0
        self.pop_size = pop_size
        self.genomes = []  # Genomes in the current population
        self.neural_nets = []  # Neural networks (phenotype) in the current population
        self.species = []  # Group similar genomes into the same species
        self.parallel = parallel
        self.best = []  # print best fitnesses for all generations TODO this is debug
        if environment is None:
            self.n_net_inputs = n_net_inputs
            self.n_net_outputs = n_net_outputs
        else:
            self.env = environment
            self.gym_env_string = gym_env_string
            self.n_net_inputs, self.n_net_outputs = get_env_spaces(gym_env_string)
        if parallel:
            import multiprocessing
            self.pool = multiprocessing.Pool(processes=processes)
        self.act_set = ActivationFunctionSet()
        self.node_set = NodeFunctionSet()
        self._get_initial_population()

    def begin_evolution(self):
        print("Starting evolution...")
        while True:  # For infinite generations
            print("Start of generation ", str(self.generation))
            self._speciate_genomes()
            print("Num of species ", len(self.species))
            parent_genomes = self._match_genomes()
            self._reproduce_and_eval_generation(parent_genomes)
            print("New generation reproduced")
            self._evaluate_population()
            self._generation_stats()
            # TODO add new links and nodes to gene pool
            print("End of generation ", str(self.generation))
            self.generation += 1

    def _speciate_genomes(self):
        """ Put genomes into species """
        global compatibility_dist
        genomes_unmatched = deque(self.genomes)
        # Put all unmatched genomes into a species or create new species if no match
        while genomes_unmatched:
            genome = genomes_unmatched.pop()
            matched = False
            # Search existing species to find match for this genome
            for s in self.species:
                if s.get_distance(genome) < compatibility_dist:
                    s.add_to_species(genome)
                    matched = True
                    break
            # No species found so create new species and use this genome as the representative genome
            if not matched:
                self.species.append(Species(genome))
        # Adjust compatibility_dist if number of species is less or more than target_num_species
        if len(self.species) < target_num_species:
            compatibility_dist -= compatibility_adjust
        elif len(self.species) > target_num_species:
            compatibility_dist += compatibility_adjust
        print("compatibility_dist ", compatibility_dist)

    def _match_genomes(self):
        """ match suitable genomes ready for reproduction """
        inds_to_reproduce = np.full(len(self.species), math.floor(self.pop_size / len(self.species)))
        inds_to_reproduce[:self.pop_size % len(self.species)] += 1
        parent_genomes = []
        # Sort genomes in each species by net fitness
        for s in self.species:
            s.genomes.sort(key=lambda genome: genome.net.fitness, reverse=True)
        # Match suitable parent genomes. Note local competition means ~equal num of genomes reproduce for each species
        for i, s in enumerate(self.species):
            for j in range(inds_to_reproduce[i]):
                if event(interspecies_mating_prob): # mate outside of species. NOTE no guarantee selected genome outside of species
                    mate_species_ind = np.random.randint(0, len(self.species))
                    mate_ind = np.random.randint(0, inds_to_reproduce[mate_species_ind])
                    parent_genomes.append((s.genomes[j], self.species[mate_species_ind].genomes[mate_ind]))
                else:  # mate within species
                    if len(s.genomes) != 1:  # For species with more than 1 genome
                        parent_genomes.append((s.genomes[j], s.genomes[np.random.randint(0, len(s.genomes))]))
                    else:  # Species only has 1 genome so copy and mutate
                        parent_genomes.append(s.genomes[j])
        return parent_genomes

    def _reproduce_and_eval_generation(self, parent_genomes):
        """ reproduce next generation given fitnesses of current generation """
        if self.parallel:
            new_genomes, new_nets, new_structures = self.pool.starmap(parallel_reproduce_eval, [(parent_genomes,
                                                                                 self.n_net_inputs,
                                                                                 self.n_net_outputs,
                                                                                 self.env,
                                                                                 self.gym_env_string) for parent_genomes in parent_genomes])
        else:
            pass
        # Add new structures to gene pool
        for structures in new_structures:
            for structure in structures:
                pass
        # Overwrite current generation genomes/nets/species TODO pickle best performing
        self.genomes = new_genomes
        self.neural_nets = new_nets

    def _generation_stats(self):
        self.neural_nets.sort(key=lambda net: net.fitness,
                              reverse=True)  # Sort nets by fitness - element 0 = fittest
        self.best.append(self.neural_nets[0].fitness_unnorm)
        print("Best fitnesses unnorm ", self.best[-100:])
        if keyboard.is_pressed('v'):
            self.env(self.gym_env_string, trials=1).evaluate(self.neural_nets[0], render=True)

    def _get_initial_population(self):
        while len(self.neural_nets) != self.pop_size:
            genome = CPPNGenome(self.gene_pool.gene_nodes_in, self.gene_pool.gene_nodes, self.gene_pool.gene_links)
            genome.create_initial_graph()
            net = Substrate().build_network_from_genome(genome, self.n_net_inputs, self.n_net_outputs)  # Express the genome to produce a neural network
            self.genomes.append(genome)
            self.neural_nets.append(net)
            print("Added genome ", len(self.genomes), " of ", self.pop_size)


