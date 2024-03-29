"""
    Control evolutionary process

    __author__ = "Joe Sarsfield"
    __email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
from genes import GenePool
from genome import CPPNGenome
import math
#from time import perf_counter  # Accurate timing
from substrate import Substrate
from environment import EnvironmentReinforcement, EnvironmentClassification, get_env_spaces
from species import Species
from config import *
from activations import ActivationFunctionSet, NodeFunctionSet
import keyboard
from evolution_worker import worker_main
import random
import ray
from collections import deque
import pickle
import datetime
import os
import hpc_initialisation
from feature_map import FeatureMap
import feature_dimensions

if __debug__:
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# TODO !!! for supervised learning envs - evolution selects initial weights and gradient-based methods optimise weights within the lifetime
# TODO !!! kill off under-performing (define under-performing) species after x (maybe 8) generations, investigate ways of introducing new random genomes

# TODO analysis of algorithms. Implement an analysis module that can determine the performance of two algorithms
#  e.g. plot the accuracy/score of algorithm a & b over x generations. Required for determining if algorithmic changes
#  are improving performance

# TODO review connection cost
# TODO investigate changing and dynamic environments
# TODO select for novelty/diversity


class Evolution:

    def __init__(self,
                 pop_size=64,
                 environment_type=EnvironmentReinforcement,
                 env_args=None,
                 session_name=None,
                 gen=None,
                 execute=Exec.PARALLEL_HPC,
                 worker_list=None,
                 persist_every_n_gens=10,
                 log_to_driver=False,
                 evaluator_callback=None,
                 feature_dims=[feature_dimensions.PerformanceDimension(feature_dimensions.fitness_dimension),
                               feature_dimensions.GenomicDimension(feature_dimensions.genome_nodes_dimension, binning=1), # feature_dimensions.PhenotypicDimension(feature_dimensions.network_links_dimension, binning=-2),
                               feature_dimensions.GenomicDimension(feature_dimensions.genome_link_cost_dimension, binning=1)]
                 ):
        """
        :param pop_size:  size of the population for each generation
        :param environment_type:  env type e.g. reinforcement or classification
        :param env_args:  arguments to pass to environment instance constructor e.g. EnvironmentReinforcement
        :param session_name: if none start new evolutionary search otherwise load evolutionary state from disk
        :param gen:  if loading then pass the generation of the session to load
        :param execute:  how is the evolutionary search being executed e.g. serially, local_parallel, hpc
        :param worker_list:  if running on multiple nodes (hpc) then pass a list of the node ip addresses for communication
        :param persist_every_n_gens: how often to persist evolutionary state to disk, -1 = never persist
        :param evaluator_callback: evaluator callback method for retrieving end of generation info. None = no evaluator
        :param feature_dims: dimensions of interest for MAP-elites algorithm (guides selection of genomes)
        """
        self.persist_every_n_gens = persist_every_n_gens  # how often should the evolutionary state be saved to disk
        self.persist_counter = 0
        self.evaluator_callback = evaluator_callback
        self.feature_dims = feature_dims
        self.feature_map = FeatureMap(feature_dims)
        self._setup_evolution(pop_size,
                              environment_type,
                              env_args,
                              session_name=session_name,
                              gen=gen)
        self.execute = execute
        if __debug__:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)
        local_mode = False
        if execute == Exec.SERIAL:
            local_mode = True
            worker_list = None
        elif execute == Exec.PARALLEL_LOCAL:
            worker_list = None
        if not ray.is_initialized():
            hpc_initialisation.initialise_hpc(worker_list, local_mode=local_mode, log_to_driver=log_to_driver)
        if session_name is None:  # create random genomes if new evolutionary search
            self._get_initial_population()

    def _setup_evolution(self,
                         pop_size,
                         environment_type,
                         env_args,
                         session_name=None,
                         gen=None):
        """ initialise or load variables """
        if session_name:  # load saved evolutionary state
            self.session_name = session_name
            self.save_dir = "./saves/" + session_name + "/"
            self.generation = gen
            self._load_evolutionary_state()
        else:  # start new evolutionary search
            self.gene_pool = GenePool(cppn_inputs=4)  # CPPN inputs x1 x2 y1 y2
            self.species = []  # Group similar genomes into the same species
            self.generation = 0
            self.pop_size = pop_size
            self.genomes = []  # Genomes in the current population
            self.compatibility_dist = compatibility_dist_init
            self.target_num_species = round(pop_size / organisms_to_species_ratio)
            #self.best = []  # print best fitnesses for all generations TODO this is debug
            #self.evolution_champs = []  # fittest genomes over all generations
            self.act_set = ActivationFunctionSet()
            self.node_set = NodeFunctionSet()
            self.env = environment_type
            if environment_type is EnvironmentReinforcement:
                self.env_args = env_args
                self.n_net_inputs, self.n_net_outputs = get_env_spaces(self.env_args[0])
            elif environment_type is EnvironmentClassification:
                global model
                import model
                self.test_features, self.test_labels, \
                self.val_features, self.val_labels, \
                self.train_features, self.train_labels = EnvironmentClassification.load_dataset(env_args[0])
                self.n_net_inputs = self.train_features.shape[-1]
                self.n_net_outputs = env_args[1]
                self.bag = model.NN_bag_model()
            else:
                self.n_net_inputs, self.n_net_outputs = 1, 1  # TODO this is debug
            if self.persist_every_n_gens != -1:
                self.session_name = str(datetime.datetime.now()).replace(" ", "_")
                self.save_dir = "~/Projects/joe/NAS-Neuroevolution//saves/" + self.session_name + "/"
                os.mkdir(self.save_dir)
                # save evolutionary search config
                with open(self.save_dir + "config" + "--" + self.session_name + ".pkl", "wb") as f:
                    pickle.dump([self.pop_size, self.target_num_species, self.act_set, self.node_set, self.env_args], f)

    def _load_evolutionary_state(self):
        """ load the state of a saved evolutionary search """
        # load config
        with open(self.save_dir + "config" + "--" + self.session_name + ".pkl", "rb") as f:
            self.pop_size, self.target_num_species, self.act_set, self.node_set, self.env_args = pickle.load(f)
        # load generation variables
        with open(self.save_dir + "gen" + str(self.generation) + "--" + self.session_name + ".pkl", "rb") as f:
            self.genomes, self.feature_map, self.feature_dims, self.gene_pool, self.compatibility_dist = pickle.load(f)

    def _save_evolutionary_state(self):
        """ save the current state of the evolutionary search to disk """
        with open(self.save_dir + "gen" + str(self.generation) + "--" + self.session_name + ".pkl", "wb") as f:
            pickle.dump([self.genomes, self.feature_map, self.feature_dims, self.gene_pool, self.compatibility_dist], f)

    def _get_initial_population(self):
        """ generate n random genomes """
        while len(self.genomes) != self.pop_size:
            genome = CPPNGenome(self.gene_pool.gene_nodes_in,
                                self.gene_pool.gene_nodes,
                                self.gene_pool.gene_links,
                                substrate_width=random.randint(1, init_substrate_width_max),
                                substrate_height=random.randint(1, init_substrate_height_max))
            genome.create_initial_graph()
            self.genomes.append(genome)
            if __debug__:
                self.logger.info("Added genome " + str(len(self.genomes)) + " of " + str(self.pop_size))
        self.parent_genomes = []
        for i in range(self.pop_size):
            self.parent_genomes.append((self.genomes[i], True))

    def begin_evolution(self):
        """ main evolution loop """
        if __debug__:
            self.logger.info("Starting evolution...")
        while True:  # For infinite generations
            if __debug__:
                self.logger.info("Start of generation " + str(self.generation))
            #self._speciate_genomes()
            if __debug__:
                self.logger.info("Num of species " + str(len(self.species)))
            #parent_genomes = self._match_genomes()
            self._reproduce_and_eval_generation(self.parent_genomes)
            self.feature_map.update_feature_map(self.genomes)
            if __debug__:
                self.logger.info("New generation reproduced")
            self._generation_stats()
            self.generation += 1
            self.parent_genomes = self.feature_map.sample_feature_map(self.pop_size)
            self._check_persist()
            if self._process_callbacks_and_stop():
                return

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
        if __debug__:
            self.logger.info("compatibility_dist " + str(self.compatibility_dist))
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
        if __debug__:
            self.logger.info("champs " + str([c.fitness for c in self.evolution_champs]))

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
        cores = int(ray.cluster_resources()["CPU"]) if self.execute is not Exec.SERIAL else 64
        nets_per_core = 3
        send_more_threshold = cores*nets_per_core
        gen_counter_start = 0
        gen_counter_end = nets_per_core
        all_genomes_sent = False
        object_ids = []
        res = []
        if self.env is EnvironmentClassification:  # bagging randomly sample 66% of dataset w/o replacement
            inds = np.random.choice(list(range(0, len(self.train_labels))), int(len(self.train_labels) * 0.66), replace=False)
            self.env_args = [self.train_features[inds], self.train_labels[inds]]
        while True:
            for core in range(cores*2):
                parents_batch = parent_genomes[gen_counter_start:gen_counter_end]
                gen_counter_start = gen_counter_end
                gen_counter_end += nets_per_core
                if len(parents_batch) == 0:
                    all_genomes_sent = True
                    break
                object_ids.extend([worker_main.remote(parents_batch,
                                                      self.n_net_inputs,
                                                      self.n_net_outputs,
                                                      self.env,
                                                      self.env_args,
                                                      self.feature_dims)])
            while True:
                object_ids_available, object_ids_not_ready = ray.wait(object_ids, timeout=1.0)
                for worker_results in ray.get(object_ids_available):
                    res.extend(worker_results)
                object_ids = list(set(object_ids) - set(object_ids_available))
                if not object_ids and all_genomes_sent:
                    break
                elif all_genomes_sent is False and len(object_ids_not_ready) < send_more_threshold:
                    if self.pop_size-len(res) < send_more_threshold:
                        nets_per_core = 1
                    break
            if all_genomes_sent:
                break
        print("GENERATION FINISHED***************************************")
        if __debug__:
            self.logger.info("execute hpc returned")
        new_genomes = []
        new_structures = []
        for r in res:
            new_genomes.append(r[0])
            new_structures.append(r[1])
        # Add new structures to gene pool
        self.gene_pool.add_new_structures(new_genomes, new_structures)
        # Overwrite current generation genomes
        self.genomes = new_genomes

    def _generation_stats(self):
        """ print gen stats when in debug or process visualisation if key pressed """
        if __debug__:
            self.logger.info("End of generation " + str(self.generation))
        if self.env is EnvironmentClassification:
            best_genomes = self.feature_map.get_fittest_genomes(n=10)
            networks = []
            for genome in best_genomes:
                genome.create_graph()
                networks.append(Substrate().build_network_from_genome(genome, self.n_net_inputs, self.n_net_outputs))
                networks[-1].init_graph()
            fitness = self.bag.predict(networks, self.test_features, self.test_labels)
            print("(test) Bagging fitness of n genomes in feature map: ", fitness)
            print("TP: ", self.bag.tp, " FN: ", self.bag.fn, " TN: ", self.bag.tn, " FP: ", self.bag.fp)
            fitness = self.bag.predict(networks, self.val_features, self.val_labels)
            print("(val) Bagging fitness of n genomes in feature map: ", fitness)
            print("TP: ", self.bag.tp, " FN: ", self.bag.fn, " TN: ", self.bag.tn, " FP: ", self.bag.fp)
            fitness = self.bag.predict(networks, self.train_features, self.train_labels)
            print("(train) Bagging fitness of n genomes in feature map: ", fitness)
            print("TP: ", self.bag.tp, " FN: ", self.bag.fn, " TN: ", self.bag.tn, " FP: ", self.bag.fp)
            for genome in best_genomes:
                genome.net = None
                genome.graph = None
        if __debug__:
            self.logger.info("Best fitnesses " + str(best["fitness"]))
        if keyboard.is_pressed('v'):
            # Visualise generation best
            self.feature_map.visualise()
            """
            best = self.feature_map.get_fittest_genomes()
            best["genome"].create_graph()
            gen_best_net = Substrate().build_network_from_genome(best["genome"], self.n_net_inputs, self.n_net_outputs)
            gen_best_net.init_graph()
            gen_best_net.visualise_neural_net()
            gen_best_net.genome.visualise_cppn()
            if self.env is EnvironmentReinforcement:
                self.env(*self.env_args, trials=1).evaluate(gen_best_net, render=True)
            gen_best_net.graph = None
            best["genome"].net = None
            """

    def _check_persist(self):
        """ check whether to persist evolutionary state to disk """
        if self.persist_every_n_gens != -1:
            self.persist_counter += 1
            if self.persist_counter == self.persist_every_n_gens:
                self.persist_counter = 0
                self._save_evolutionary_state()

    def _process_callbacks_and_stop(self):
        """ process any callbacks and check if evaluator stopping condition is met, True = stop evaluating """
        if self.evaluator_callback is not None:
            self.genomes.sort(key=lambda genome: genome.fitness, reverse=True)  # Sort nets by fitness - element 0 = fittest
            return self.evaluator_callback(self.generation, self.genomes[0].fitness)  # pass generation info to evaluator callback
        return False

