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
import igraph
from scipy.spatial import distance


class Dimension:

    def __init__(self, binning=0):
        self.metric = -9999
        self.binning = binning


class PerformanceDimension(Dimension):
    """  performance/fitness dimensions e.g. distance walked by biped, energy used by biped """

    def __init__(self, calc_metric_func, binning=3):
        super().__init__(binning=binning)
        self.calc_metric_func = calc_metric_func

    def call(self, network):
        self.metric = np.around(self.calc_metric_func(network), self.binning).astype(np.float)


class PhenotypicDimension(Dimension):
    """ phenotypic feature dimensions i.e. neural network features e.g. num of links, nodes and network modularity """

    def __init__(self, calc_metric_func, binning=3):
        super().__init__(binning=binning)
        self.calc_metric_func = calc_metric_func

    def call(self, network):
        self.metric = np.around(self.calc_metric_func(network), self.binning).astype(np.float)


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
    return len(network.links)


def network_link_cost_dimension(network):
    """ Connection cost """
    total_dist = 0
    for l in network.links:
        total_dist += distance.euclidean((l.in_node.x, l.in_node.y), (l.out_node.x, l.out_node.y))
    return total_dist / len(network.links)


def network_link_costx_dimension(network):
    """ Connection cost """
    total_dist = 0
    for l in network.links:
        total_dist += abs(l.in_node.x - l.out_node.x)
    return total_dist / len(network.links)


def network_link_costy_dimension(network):
    """ Connection cost """
    total_dist = 0
    for l in network.links:
        total_dist += abs(l.in_node.y - l.out_node.y)
    return total_dist / len(network.links)


def network_nodes_dimension(network):
    """ Number of nodes """
    return len(network.nodes)


def network_modularity_dimension(network):
    """ calculate the modularity of the neural network """
    import random
    #if random.randint(1, 256) == 1:
    g = igraph.Graph(directed=True)
    nodes = network.input_nodes + network.nodes
    links = [(link.out_node.node_ind, link.in_node.node_ind) for link in network.links]
    #nodes = list(set(sum(links, ())))
    #layout = g.layout("layout_drl")
    g.add_vertices(len(nodes))
    """
    for i, v in enumerate(g.vs):
        v["x"] = nodes[i].x
        v["y"] = nodes[i].y
    """
    g.add_edges(links)
    comms = g.community_infomap()
    return comms.q
    #if comms.modularity == 0:
    #igraph.plot(comms, mark_groups=True)
    #igraph.plot(g, vertex_color=[color_list[x] for x in community.membership])
    #print("")


def biped_symmetry_dimension():
    """ for biped walking environments, quantity measuring the symmetry of the legs """
    pass