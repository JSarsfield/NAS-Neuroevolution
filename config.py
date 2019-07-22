"""
Configuration hyperparameters for balancing evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import random


# TODO allow loading params from yaml config
weight_max_value = 2
weight_min_value = -2
weight_mutate_rate = 0.8  # Chance of link weight being adjusted by value drawn from zero-centered normal distribution
weight_replace_rate = 0.05  # Chance of link weight being replaced with random value

bias_max_value = 2
bias_min_value = -2
bias_mutate_rate = 0.8
bias_replace_rate = 0.05

link_toggle_prob = 0.05  # Chance of link being toggled between enabled/disabled
link_add_prob = 0.1
node_add_prob = 0.05
link_enable_prob = 0.2  # Chance of disabled link being re-enabled

change_act_prob = 0.05  # Chance of changing activation function to random act func

elitism_thresh = 1  # Number of fittest organisms in each species that is preserved as-is to next generation
pop_survival_thresh = 0.2  # Fraction of population that is allowed to reproduce for next generation
min_species_size = 2  # min number of organisms/nets in a species

compatibility_thresh = 3  # Max distance two genomes can be considered as the same species TODO needs to be adaptive
compatibility_excess_coeff = 1  # Balance the distance calculation against weights and disjoint genes
compatibility_disjoint_coeff = 1  # Balance the distance calculation against weights and excess genes
compatibility_weight_coeff = 2  # Balance the distance calculation against excess and disjoint genes

gauss_weight_scale = 0.1  # Scale of gaussian function for adjusting gene link weights

target_num_species = 10  # Number of species to target TODO if species grows above this increase compatibility_thresh to reduce species

diversity_coeff = 1  # increase/decrease the level of mutation based on species sizes and species ages

interspecies_mating_prob = -0.01  # Chance of genome mating outside of species TODO changed value to minus - undo this


def event(x):
    """ determine whether an event occurred given probability coefficient """
    return False if random.random() > x else True
