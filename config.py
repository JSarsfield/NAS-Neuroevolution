"""
Configuration hyperparameters for balancing evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import random
from enum import Enum


# TODO allow loading params from yaml config
# weight params
weight_max_value = 2
weight_min_value = -2
weight_mutate_rate = 0.2  # Chance of link weight being adjusted by value drawn from zero-centered normal distribution
weight_replace_rate = 0.002  # Chance of link weight being replaced with random value
weight_init_min = -2  # min value of weight initialisation range
weight_init_max = 2  # max value of weight initialisation range
uniform_weight_scale = 0.1  # Scale of gaussian function for adjusting gene link weights
# bias params  / phase shift into act func
bias_max_value = 5
bias_min_value = -5
bias_mutate_rate = 0.1
bias_replace_rate = 0.002
bias_init_min = -2  # min value of bias initialisation range
bias_init_max = 2  # max value of bias initialisation range
# structural params
link_toggle_prob = 0.05  # Chance of link being toggled between enabled/disabled
link_add_prob = 0.1  # Chance of adding a new link
link_add_attempts = 10  # Number of attempts to find new link until give up
node_add_prob = 0.05
link_enable_prob = 0.2  # Chance of disabled link being re-enabled
new_link_attempts = 10  # How many attempts should we try and find a new node before giving up
change_act_prob = 0.05  # Chance of changing activation function to random act func
# act func coefficients
func_adjust_prob = 0.02
guass_freq_adjust = 0.02  # adjust function frequency
sin_freq_adjust = 0.02
func_amp_adjust = 0.02  # adjust function amplitude
func_vshift_adjust = 0.02  # adjust function vertical shift
func_amp_range = 0.5  # init ranges
gauss_freq_range = 0.5
sin_freq_range = 6
gauss_vshift_range = 0.25
sin_vshift_range = 1
# es-hyperneat params
init_var_thresh = 0.3
init_band_thresh = 0
var_mutate_prob = 0.05
band_mutate_prob = 0.05
#gauss_var_scale = 0.00001  # Scale of gaussian function for adjusting QuadTree variance threshold
#gauss_band_scale = 0.00001  # Scale of gaussian function for adjusting QuadTree band pruning threshold
quad_tree_max_depth = 10  # The max depth the quadtree will split if variance is still above variance threshold
substrate_search_max_time = 10  # max num of seconds to search for hid nodes on substrate before giving up and marking net as void
# substrate params
init_substrate_width = 8
init_substrate_height = 4
width_mutate_prob = 0.05
height_mutate_prob = 0.05
# reproduce params
compatibility_dist_init = 2  # Max distance two genomes can be considered as the same species
compatibility_adjust = 0.1  # Amount to adjust compatibility_dist each gen to achieve target num species
compatibility_excess_coeff = 1  # Balance the distance calculation against weights and disjoint genes
compatibility_disjoint_coeff = 2  # Balance the distance calculation against weights and excess genes
compatibility_weight_coeff = 2  # Balance the distance calculation against excess and disjoint genes
target_num_species = 25  # Number of species to target
species_survival_thresh = 0.3  # Fraction of species that is allowed to reproduce for next generation
interspecies_mating_prob = 0.15  # Chance of genome mating outside of species
genome_crossover_prob = 0.2  # chance of crossover with another genome instead of copy with mutation

diversity_coeff = 1  # increase/decrease the level of mutation based on species sizes and species ages

select_diverse_champs_prob = 0.1  # The probability of filling champs based on max genomic distance (encourages diversity)
num_evolution_champs = 15  # Number of all time champs to put back into the population after each generation
link_cost_coeff = 0.01  # coefficient cost for adjusting fitness based on number of links/connections in neural network
#elitism_thresh = 1  # Number of fittest organisms in each species that is preserved as-is to next generation
#min_species_size = 2  # min number of organisms/nets in a species


def event(x):
    """ determine whether an event occurred given probability coefficient """
    return False if random.random() > x else True


class Exec(Enum):
    SERIAL = 1
    PARALLEL_LOCAL = 2
    PARALLEL_HPC = 3
