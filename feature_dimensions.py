"""
    Feature dimension functions used in MAP-elites

    A feature map needs to initially define the feature dimension functions for the environment in this module.
    The dimensions of interest funcs can then be saved to a feature_map file for the next time the environment is run.

    __author__ = "Joe Sarsfield"
    __email__ = "joe.sarsfield@gmail.com"
"""


def gym_fitness_dimension():
    """ fitness dimension for gym environments """
    pass


def network_size_dimension():
    """ neural network size. Number of links + number of nodes """
    # todo consider calculating size as sum of length of links
    pass


def biped_symmetry_dimension():
    """ for biped walking environments, quantity measuring the symmetry of the legs """
    pass