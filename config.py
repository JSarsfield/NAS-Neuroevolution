"""
Configuration hyperparameters for balancing evolutionary process

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""


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

elitism_thresh = 1  # Number of fittest organisms in each species that is preserved as-is to next generation
species_survival_thresh = 0.2  # Fraction of each species that is allowed to reproduce for each generation
min_species_size = 2  # min number of organisms/nets in a species

compatibility_thresh = 2  # Max distance two genomes can be considered as the same species
compatibility_excess_coeff = 1  # Balance the distance calculation against weights and disjoint genes
compatibility_disjoint_coeff = 1  # Balance the distance calculation against weights and excess genes
compatibility_weight_coeff = 1  # Balance the distance calculation against excess and disjoint genes

diversity_coeff = 1  # increase/decrease the level of mutation based on species sizes and species ages