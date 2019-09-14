"""
Control evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
from genes import GenePool
from genome import CPPNGenome
from collections import deque
import math
#from time import perf_counter  # Accurate timing
from substrate import Substrate
from environment import EnvironmentReinforcement, get_env_spaces
from species import Species
from config import *
from activations import ActivationFunctionSet, NodeFunctionSet
import keyboard
from evolution_parallel import parallel_reproduce_eval
import random
import ray
from collections import deque

# TODO !!! for supervised learning envs remove weights from evolution and optimise within the lifetime
# TODO !!! kill off under-performing species after x (maybe 8) generations, investigate ways of introducing new random genomes

# TODO analysis of algorithms. Implement an analysis module that can determine the performance of two algorithms
#  e.g. plot the accuracy/score of algorithm a & b over x generations. Required for determining if algorithmic changes
#  are improving performance

# TODO pickle top performing genomes after each/x generations
# TODO review connection cost
# TODO investigate changing and dynamic environments
# TODO select for novelty/diversity


class Evolution:

    def __init__(self, n_net_inputs,
                 n_net_outputs,
                 pop_size=10,
                 environment=None,
                 gym_env_string="BipedalWalker-v2",
                 yaml_config=None,
                 execute=Exec.PARALLEL_HPC,
                 worker_list=None,
                 processes=64):
        self.gene_pool = GenePool(cppn_inputs=4)  # CPPN inputs x1 x2 y1 y2
        self.generation = 0
        self.pop_size = pop_size
        self.genomes = []  # Genomes in the current population
        self.species = []  # Group similar genomes into the same species
        self.execute = execute
        self.best = []  # print best fitnesses for all generations TODO this is debug
        self.evolution_champs = []  # fittest genomes over all generations
        self.compatibility_dist = compatibility_dist_init
        self.target_num_species = round(pop_size/organisms_to_species_ratio)
        if environment is None:
            self.n_net_inputs = n_net_inputs
            self.n_net_outputs = n_net_outputs
        else:
            self.env = environment
            self.gym_env_string = gym_env_string
            self.n_net_inputs, self.n_net_outputs = get_env_spaces(gym_env_string)
        if execute == Exec.PARALLEL_LOCAL:
            from evolution_parallel import parallel_reproduce_eval
            import multiprocessing
            self.pool = multiprocessing.Pool(processes=processes)
        elif execute == Exec.PARALLEL_HPC:
            global hpc_initialisation
            import hpc_initialisation
            hpc_initialisation.initialise_hpc(worker_list)
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
        if len(self.species) < self.target_num_species:
            self.compatibility_dist -= compatibility_adjust
        elif len(self.species) > self.target_num_species:
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
        if self.execute == Exec.SERIAL:
            pass
        elif self.execute == Exec.PARALLEL_LOCAL:
            res = self.pool.starmap(parallel_reproduce_eval, [(parent,
                                                               self.n_net_inputs,
                                                               self.n_net_outputs,
                                                               self.env,
                                                               self.gym_env_string) for parent in parent_genomes])
        else:  # Exec.PARALLEL_HPC
            res = ray.get([parallel_reproduce_eval.remote(parent, self.n_net_inputs, self.n_net_outputs, self.env, self.gym_env_string) for parent in parent_genomes])
            """
            while True:
                objects_ids = deque([])
                for i, parent in enumerate(parent_genomes):
                    objects_ids.append(parallel_reproduce_eval.remote(parent, self.n_net_inputs, self.n_net_outputs, self.env, self.gym_env_string))
                res = ray.get([objects_ids.popleft() for _ in range(10)])
            """
        print("execute hpc returned")
        new_genomes = []
        new_structures = []
        for r in res:
            new_genomes.append(r[0])
            new_structures.append(r[1])
        # Add new structures to gene pool
        self.gene_pool.add_new_structures(new_genomes, new_structures)
        # Overwrite current generation genomes TODO pickle best performing
        self.genomes = new_genomes

    def _generation_stats(self):
        self.genomes.sort(key=lambda genome: genome.fitness,
                              reverse=True)  # Sort nets by fitness - element 0 = fittest
        self.best.append(self.genomes[0].fitness)
        print("Best fitnesses ", self.best[-100:])
        if keyboard.is_pressed('v'):
            # Visualise generation best
            gen_best_net = Substrate().build_network_from_genome(self.genomes[0], self.n_net_inputs, self.n_net_outputs)
            gen_best_net.init_graph()
            gen_best_net.visualise_neural_net()
            gen_best_net.genome.visualise_cppn()
            self.env(self.gym_env_string, trials=1).evaluate(gen_best_net, render=True)
            gen_best_net.graph = None
            self.genomes[0].net = None
            # Visualise overall best (champ)
            if self.evolution_champs[0].graph is None:
                self.evolution_champs[0].create_graph()
            champ_net = Substrate().build_network_from_genome(self.evolution_champs[0], self.n_net_inputs, self.n_net_outputs)
            if champ_net.is_void is False:
                champ_net.init_graph()
                champ_net.visualise_neural_net()
                champ_net.genome.visualise_cppn()
                self.env(self.gym_env_string, trials=1).evaluate(champ_net, render=True)
            champ_net.graph = None
            self.evolution_champs[0].net = None
            self.evolution_champs[0].graph = None

    def _get_initial_population(self):
        while len(self.genomes) != self.pop_size:
            genome = CPPNGenome(self.gene_pool.gene_nodes_in,
                                self.gene_pool.gene_nodes,
                                self.gene_pool.gene_links,
                                substrate_width=random.randint(1, init_substrate_width_max),
                                substrate_height=random.randint(1, init_substrate_height_max))
            genome.create_initial_graph()
            self.genomes.append(genome)
            print("Added genome ", len(self.genomes), " of ", self.pop_size)


