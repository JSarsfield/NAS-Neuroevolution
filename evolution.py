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
from environment import EnvironmentReinforcement, get_env_spaces
from species import Species
from config import *
from genes import GeneLink, GeneNode
from activations import ActivationFunctionSet, NodeFunctionSet
import keyboard
import copy
import operator

# TODO !!! kill off under-performing species after x (maybe 8) generations, investigate ways of introducing new random genomes

# TODO analysis of algorithms. Implement an analysis module that can determine the performance of two algorithms
#  e.g. plot the accuracy/score of algorithm a & b over x generations. Required for determining if algorithmic changes
#  are improving performance

# TODO pickle top performing genomes after each/x generations
# TODO review connection cost
# TODO investigate changing and dynamic environments
# TODO select for novelty/diversity


def parallel_reproduce_eval(parent_genomes, n_net_inputs, n_net_outputs, env, gym_env_string):
    # Reproduce from parent genomes
    if type(parent_genomes[-1]) is not bool:  # Two parent genomes so crossover
        genome, new_structures = crossover(parent_genomes[0], parent_genomes[1])
    else:  # One parent genome so mutate
        genome, new_structures = copy_with_mutation(parent_genomes[0], mutate=parent_genomes[1])
    # Create genome graph
    genome.create_graph()
    # Create net from genome
    net = Substrate().build_network_from_genome(genome, n_net_inputs, n_net_outputs)
    if not net.is_void:
        net.init_graph()  # init TF graph
        # Evaluate
        fitness = env(gym_env_string).evaluate(net)
        net.set_fitness(fitness)
        net.graph = None  # TF graph can't be pickled so delete it
    return (genome, net, new_structures)


def crossover(g1, g2):
    """ crossover of two parent genomes """
    i = 0
    j = 0
    nodes_to_add = []
    links_to_add = []
    sub_width = g1.substrate_width if g1.fitness >= g2.fitness else g2.substrate_width
    sub_height = g1.substrate_height if g1.fitness >= g2.fitness else g2.substrate_height
    while i < len(g1.gene_links) or j < len(g2.gene_links):
        if i < len(g1.gene_links) and j < len(g2.gene_links):
            if g1.gene_links[i].historical_marker == g2.gene_links[j].historical_marker:
                if g1.fitness >= g2.fitness:
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
    return create_new_genome(nodes_to_add, links_to_add, g1.cppn_inputs, sub_width, sub_height)


def copy_with_mutation(g1, mutate=True):
    """ copy a genome with mutation """
    return create_new_genome(g1.gene_nodes_in + g1.gene_nodes,
                             g1.gene_links,
                             g1.cppn_inputs,
                             g1.substrate_width,
                             g1.substrate_height,
                             mutate=mutate)


def create_new_genome(nodes_to_add, links_to_add, cppn_inputs, sub_width, sub_height, mutate=True):
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
    if mutate:
        new_structures = mutate_structural(gene_nodes, gene_links)  # This is performed on master thread to ensure only new genes are added to gene pool
    else:
        new_structures = []
    gene_nodes_in = gene_nodes[:cppn_inputs]
    gene_nodes = gene_nodes[cppn_inputs:]
    new_genome = CPPNGenome(gene_nodes_in, gene_nodes, gene_links, substrate_width=sub_width, substrate_height=sub_height)
    if mutate:
        new_genome.mutate_nonstructural()
    return new_genome, new_structures


def get_node_from_hist_marker(gene_nodes, hist_marker):
    for node in gene_nodes:
        if node.historical_marker == hist_marker:
            return node
    raise Exception("No node with historical marker found in func get_node_from_hist_marker evolution.py")


def mutate_structural(gene_nodes, gene_links):
    """ mutate genome to add nodes and links. Note passed by reference """
    # TODO !!! allow nodes to become disabled and thus all ingoing out going links disabled
    # TODO add config probs for toggling NODE! enable/disable
    new_structures = {}  # keep track of new links and nodes to ensure correct historical marker is applied after
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
                    #new_link = gene_pool.get_or_create_gene_link(gene_nodes[in_node_ind].historical_marker, out_node.historical_marker)
                    gene_links.append((random.uniform(weight_init_min, weight_init_max),
                                           gene_nodes[in_node_ind].historical_marker,
                                           out_node.historical_marker,
                                           None,
                                           True))
                    new_structures["new_link"] = (gene_links[-1][1], gene_links[-1][2])
                    break
                attempt += 1
            if attempt == new_link_attempts:
                break
    # Mutate add node with random activation function
    if event(node_add_prob):
        # Get a random link to split and add node
        gene_link_ind = random.randint(0, len(gene_links)-1)
        old_link = gene_links[gene_link_ind]
        in_node = get_node_from_hist_marker(gene_nodes, old_link[1])
        out_node = get_node_from_hist_marker(gene_nodes, old_link[2])
        old_link_update = (old_link[0], old_link[1], old_link[2], old_link[3], False)
        del gene_links[gene_link_ind]
        gene_links.append(old_link_update)
        act_set = ActivationFunctionSet()
        node_set = NodeFunctionSet()
        # Create new node
        new_structures["new_node"] = (old_link[1], old_link[2])
        new_structures["node_depth"] = out_node.depth+((in_node.depth-out_node.depth)/2)
        gene_nodes.append(GeneNode(new_structures["node_depth"],
                                   act_set.get_random_activation_func(),
                                   node_set.get("dot"),
                                   None,
                                   True,
                                   True,
                                   random.uniform(bias_init_min, bias_init_max)))
        # Create new link going into new node
        gene_links.append((1,
                           in_node.historical_marker,
                           gene_nodes[-1].historical_marker,
                           None,
                           True))
        # Create new link going out of new node
        gene_links.append((old_link[0],
                           gene_nodes[-1].historical_marker,
                           out_node.historical_marker,
                           None,
                           True))
    gene_nodes.sort(key=lambda x: x.depth)
    return new_structures


class Evolution:

    def __init__(self, n_net_inputs,
                 n_net_outputs,
                 pop_size=10,
                 environment=None,
                 gym_env_string="BipedalWalker-v2",
                 yaml_config=None,
                 parallel=True,
                 processes=64):
        self.gene_pool = GenePool(cppn_inputs=4)  # CPPN inputs x1 x2 y1 y2
        self.generation = 0
        self.pop_size = pop_size
        self.genomes = []  # Genomes in the current population
        self.neural_nets = []  # Neural networks (phenotype) in the current population
        self.species = []  # Group similar genomes into the same species
        self.parallel = parallel
        self.best = []  # print best fitnesses for all generations TODO this is debug
        self.evolution_champs = []  # fittest genomes over all generations
        self.compatibility_dist = compatibility_dist_init
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
            self._generation_stats()
            print("End of generation ", str(self.generation))
            self.generation += 1

    def _speciate_genomes(self):
        """ Put genomes into species """
        self.species = []
        for genome in self.evolution_champs:
            self.genomes.append(CPPNGenome(genome.gene_nodes_in,
                                           genome.gene_nodes,
                                           genome.gene_links,
                                           substrate_width=genome.substrate_width,
                                           substrate_height=genome.substrate_height,
                                           fitness=genome.fitness))  # Add best genome from all generations
        genomes_unmatched = deque(self.genomes)
        # Put all unmatched genomes into a species or create new species if no match
        while genomes_unmatched:
            genome = genomes_unmatched.pop()
            matched = False
            # Search existing species to find match for this genome
            for s in self.species:
                if s.get_distance(genome) < self.compatibility_dist:
                    s.add_to_species(genome)
                    matched = True
                    break
            # No species found so create new species and use this genome as the representative genome
            if not matched:
                self.species.append(Species(genome))
        # Adjust compatibility_dist if number of species is less or more than target_num_species
        if len(self.species) < target_num_species:
            self.compatibility_dist -= compatibility_adjust
        elif len(self.species) > target_num_species:
            self.compatibility_dist += compatibility_adjust
        print("compatibility_dist ", self.compatibility_dist)
        # Sort species and champs
        for s in self.species:
            s.genomes.sort(key=lambda x: x.fitness, reverse=True)
        self.species.sort(key=lambda x: x.genomes[0].fitness, reverse=True)  # Sort species by fittest genome in species
        self.evolution_champs.sort(key=lambda genome: genome.fitness, reverse=True)
        # Cull champs
        if len(self.evolution_champs) > len(self.species):
            self.evolution_champs = self.evolution_champs[:len(self.species)]
        # Add champs
        elif len(self.evolution_champs) < len(self.species):
            # find genomes that are furthest away from the other champs (encourage diversity)
            dists = []
            if event(select_diverse_champs_prob):  # diverge by selecting best genomes from species with max genomic dist
                for i in range(len(self.species)):
                    dists.append((i, sum([self.species[i].get_distance(c) for c in self.evolution_champs])))
                dists.sort(key=lambda x: x[1], reverse=True)
            else:  # Add best genomes from best performing species
                for i in range(0, len(self.species)-len(self.evolution_champs)):
                    dists.append([i])
            for i in range(0, len(self.species)-len(self.evolution_champs)):
                self.evolution_champs.append(CPPNGenome(self.species[dists[i][0]].genomes[0].gene_nodes_in, # TODO consider overriding genome copy instead
                                                        self.species[dists[i][0]].genomes[0].gene_nodes,
                                                        self.species[dists[i][0]].genomes[0].gene_links,
                                                        substrate_width=self.species[dists[i][0]].genomes[0].substrate_width,
                                                        substrate_height=self.species[dists[i][0]].genomes[0].substrate_height,
                                                        fitness=self.species[dists[i][0]].genomes[0].fitness))
        # Replace champs with closest genome that is fitter
        for i in range(len(self.species)):
            ind, _ = min(enumerate(self.evolution_champs), key=lambda champ: self.species[i].get_distance(champ[1]))
            # Replace if species best genome is fitter than closest champ genome
            if self.species[i].genomes[0].fitness > self.evolution_champs[ind].fitness:
                self.evolution_champs[ind] = CPPNGenome(self.species[i].genomes[0].gene_nodes_in,
                                                        self.species[i].genomes[0].gene_nodes,
                                                        self.species[i].genomes[0].gene_links,
                                                        substrate_width=self.species[i].genomes[0].substrate_width,
                                                        substrate_height=self.species[i].genomes[0].substrate_height,
                                                        fitness=self.species[i].genomes[0].fitness)
        print("champs ", [c.fitness for c in self.evolution_champs])

    def _match_genomes(self):
        """ match suitable genomes ready for reproduction """
        inds_to_reproduce = np.full(len(self.species), math.floor(self.pop_size / len(self.species)))
        inds_to_reproduce[:self.pop_size % len(self.species)] += 1
        parent_genomes = []
        # Sort genomes in each species by net fitness
        for s in self.species:
            s.genomes.sort(key=lambda genome: genome.fitness, reverse=True)
        # Match suitable parent genomes. Note local competition means ~equal num of genomes reproduce for each species
        for i, s in enumerate(self.species):
            j = 0  # index of genomes in species that are allowed to reproduce
            stop_ind = math.ceil(len(s.genomes) * species_survival_thresh)  # j resets to 0 when equal to stop_ind
            for k in range(inds_to_reproduce[i]):
                if event(interspecies_mating_prob): # mate outside of species. NOTE no guarantee selected genome outside of species
                    mate_species_ind = np.random.randint(0, len(self.species))
                    mate_ind = np.random.randint(0, math.ceil(len(self.species[mate_species_ind].genomes) * species_survival_thresh))
                    parent_genomes.append((s.genomes[j], self.species[mate_species_ind].genomes[mate_ind]))
                else:  # mate within species
                    if event(genome_crossover_prob) and len(s.genomes) != 1:  # For species with more than 1 genome
                        parent_genomes.append((s.genomes[j], s.genomes[np.random.randint(0, stop_ind)]))
                    else:  # Species only has 1 genome so copy and mutate
                        if k == 0:
                            parent_genomes.append((s.genomes[j], False))  # Copy species winner without mutation
                        else:
                            parent_genomes.append((s.genomes[j], True))
                j = 0 if j == stop_ind-1 else j+1
        return parent_genomes

    def _reproduce_and_eval_generation(self, parent_genomes):
        """ reproduce next generation given fitnesses of current generation """
        if self.parallel:
            res = self.pool.starmap(parallel_reproduce_eval, [(parent_genomes,
                                                               self.n_net_inputs,
                                                               self.n_net_outputs,
                                                               self.env,
                                                               self.gym_env_string) for parent_genomes in parent_genomes])
            new_genomes = []
            new_nets = []
            new_structures = []
            for r in res:
                new_genomes.append(r[0])
                new_nets.append(r[1])
                new_structures.append(r[2])
        else:
            pass
        # Add new structures to gene pool
        self.gene_pool.add_new_structures(new_genomes, new_structures)
        # Overwrite current generation genomes/nets/species TODO pickle best performing
        self.genomes = new_genomes
        self.neural_nets = new_nets

    def _generation_stats(self):
        self.neural_nets.sort(key=lambda net: net.fitness,
                              reverse=True)  # Sort nets by fitness - element 0 = fittest
        self.best.append(self.neural_nets[0].fitness)
        print("Best fitnesses ", self.best[-100:])
        if keyboard.is_pressed('v'):
            self.neural_nets[0].visualise_neural_net()
            self.neural_nets[0].genome.visualise_cppn()
            self.neural_nets[0].init_graph()
            self.env(self.gym_env_string, trials=1).evaluate(self.neural_nets[0], render=True)
            self.neural_nets[0].graph = None

    def _get_initial_population(self):
        while len(self.neural_nets) != self.pop_size:
            genome = CPPNGenome(self.gene_pool.gene_nodes_in,
                                self.gene_pool.gene_nodes,
                                self.gene_pool.gene_links,
                                substrate_width=init_substrate_width,
                                substrate_height=init_substrate_height)
            genome.create_initial_graph()
            net = Substrate().build_network_from_genome(genome, self.n_net_inputs, self.n_net_outputs)  # Express the genome to produce a neural network
            self.genomes.append(genome)
            self.neural_nets.append(net)
            print("Added genome ", len(self.genomes), " of ", self.pop_size)


