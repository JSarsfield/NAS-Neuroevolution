"""
    Feature dimension functions used in MAP-elites

    A feature map needs to initially define the feature dimension functions for the environment in this module.
    The dimensions of interest funcs can then be saved to a feature_map file for the next time the environment is run.


    Types of feature dimensions:
    1. performance/fitness dimensions e.g. distance walked by biped, energy used by biped
    2. action dimensions e.g. frequency of a type of action, quantity of the action
    3. observation/behaviour dimensions e.g. symmetry in biped gait
    4. phenotypic feature dimensions i.e. neural network features e.g. number of links, nodes and network modularity

    A list of these dimensions exist for a given env and the metrics are calculated for each network in parallel.
    A single feature map exists on the master process that updates the feature maps and selects parent genomes for crossover

    __author__ = "Joe Sarsfield"
    __email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np


class Dimension:

    def __init__(self, binning=0):
        self.metric = -9999
        self.binning = binning


class PerformanceDimension(Dimension):
    """  performance/fitness dimensions e.g. distance walked by biped, energy used by biped """

    def __init__(self, calc_metric_func):
        super().__init__()
        self.calc_metric_func = calc_metric_func

    def call(self, network):
        self.metric = np.around(self.calc_metric_func(network), self.binning).astype(np.int)


class PhenotypicDimension(Dimension):
    """ phenotypic feature dimensions i.e. neural network features e.g. num of links, nodes and network modularity """

    def __init__(self, calc_metric_func):
        super().__init__()
        self.calc_metric_func = calc_metric_func

    def call(self, network):
        self.metric = np.around(self.calc_metric_func(network), self.binning).astype(np.int)


class ActionDimension(Dimension):
    """ action dimensions e.g. frequency of a type of action, quantity of the action """

    def __init__(self):
        super().__init__()


class BehaviourDimension(Dimension):
    """ observation/behaviour dimensions e.g. symmetry in biped gait """

    def __init__(self):
        super().__init__()


def fitness_dimension(network):
    """ fitness dimension e.g. fitness from gym envs """
    return network.fitness


def network_links_dimension(network):
    """ Number of links """
    # todo consider calculating size as sum of length of links
    return len(network.links)


def network_nodes_dimension(network):
    """ Number of nodes """
    return len(network.nodes)


def network_modularity_dimension(network):
    """ calculate the modularity of the neural network """
    m = len(network.links)
    m2 = m*2
    norm = 1/m2


def biped_symmetry_dimension():
    """ for biped walking environments, quantity measuring the symmetry of the legs """
    pass