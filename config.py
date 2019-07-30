"""
Configuration hyperparameters for balancing evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import random


# TODO allow loading params from yaml config
# weight params
weight_max_value = 2
weight_min_value = -2
weight_mutate_rate = 0.8  # Chance of link weight being adjusted by value drawn from zero-centered normal distribution
weight_replace_rate = 0.005  # Chance of link weight being replaced with random value
weight_init_min = -1  # min value of weight initialisation range
weight_init_max = 1  # max value of weight initialisation range
gauss_weight_scale = 0.07  # Scale of gaussian function for adjusting gene link weights
# bias params
bias_max_value = 1
bias_min_value = -1
bias_mutate_rate = 0.8
bias_replace_rate = 0.05
bias_init_min = -0.1  # min value of bias initialisation range
bias_init_max = 0.1  # max value of bias initialisation range
# structural params
link_toggle_prob = 0.05  # Chance of link being toggled between enabled/disabled
link_add_prob = 0.08  # Chance of adding a new link # TODO set back to 0.08
link_add_attempts = 10  # Number of attempts to find new link until give up
node_add_prob = 0.03  # TODO set back to 0.03
link_enable_prob = 0.2  # Chance of disabled link being re-enabled
new_link_attempts = 10  # How many attempts should we try and find a new node before giving up
# es-hyperneat params
init_var_thresh = 0.3
init_band_thresh = 0
var_mutate_prob = 0.1
band_mutate_prob = 0.1
gauss_var_scale = 0.07  # Scale of gaussian function for adjusting QuadTree variance threshold
gauss_band_scale = 0.07  # Scale of gaussian function for adjusting QuadTree band pruning threshold
quad_tree_max_depth = 10  # The max depth the quadtree will split if variance is still above variance threshold


substrate_search_max_time = 20  # max num of seconds to search for hid nodes on substrate before giving up and marking net as void

change_act_prob = 0.05  # Chance of changing activation function to random act func

elitism_thresh = 1  # Number of fittest organisms in each species that is preserved as-is to next generation
pop_survival_thresh = 0.2  # Fraction of population that is allowed to reproduce for next generation
min_species_size = 2  # min number of organisms/nets in a species

compatibility_thresh = 3  # Max distance two genomes can be considered as the same species TODO needs to be adaptive
compatibility_excess_coeff = 2  # Balance the distance calculation against weights and disjoint genes
compatibility_disjoint_coeff = 2  # Balance the distance calculation against weights and excess genes
compatibility_weight_coeff = 1  # Balance the distance calculation against excess and disjoint genes


target_num_species = 10  # Number of species to target TODO if species grows above this increase compatibility_thresh to reduce species

diversity_coeff = 1  # increase/decrease the level of mutation based on species sizes and species ages

interspecies_mating_prob = 0.01  # Chance of genome mating outside of species


def event(x):
    """ determine whether an event occurred given probability coefficient """
    return False if random.random() > x else True
