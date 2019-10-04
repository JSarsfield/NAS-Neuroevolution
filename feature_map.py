"""
    Feature/behaviour map as defined in map-elites.

    __author__ = "Joe Sarsfield"
    __email__ = "joe.sarsfield@gmail.com"
"""
import numpy as np
from scipy.spatial import KDTree  # Find nearest neighbours
from config import *
import random


class FeatureMap:
    """ feature map containing feature dimensions of interest, including performance dims,
     phenotypic dims, action dims and observation/behaviour dims """

    # TODO introduce a resolution to the dimensions that increases over time
    # TODO consider changing the feature dims or have multiple feature maps
    # TODO select better feature dims based on modularity and link squared distance

    def __init__(self, feature_dims, load_map=None):
        self.feature_dims = feature_dims
        self.coords_to_genome = {}  # get genome in cell given coordinates of cell

    def dimensions_of_interest(self):
        """ feature/behaviour of interest, multiple dimensions can exist within the map """
        pass

    def performance_dimension(self):
        """ every map has atleast one performance/fitness metric dimension """
        pass

    def update_feature_map(self, genomes):
        """ updates cells of feature map if genome performance is highest in cell or cell empty """
        for genome in genomes:
            if genome.phenotypic_dims is None:
                continue
            coord = []
            for dim in genome.phenotypic_dims:
                coord.append(dim.metric)
            coord = tuple(coord)
            if coord in self.coords_to_genome:  # genome already exists in cell so check if performance is greater
                if genome.fitness >= self.coords_to_genome[coord]["fitness"]:
                    self.coords_to_genome[coord] = {"genome": genome, "fitness": genome.fitness}
            else:  # no genome occupying cell so add it
                self.coords_to_genome[coord] = {"genome": genome, "fitness": genome.fitness}

    def sample_feature_map(self, n_samples):
        """ sample n genomes from random cells in the feature map """
        # TODO consider dynamic bias for selecting cells for crossover
        parent_genomes = []
        tree = KDTree(list(self.coords_to_genome.keys()))
        while len(parent_genomes) < n_samples:
            genome_keys = random.sample(self.coords_to_genome.keys(), k=min(len(self.coords_to_genome.keys()), n_samples))
            for key in genome_keys:
                if event(genome_crossover_prob):  # crossover
                    neighbours = tree.query([key], crossover_neighbour_elites)
                    parent_genomes.append((self.coords_to_genome[key]["genome"],
                                           self.coords_to_genome[tuple(tree.data[random.choice(tree.query([key], crossover_neighbour_elites)[1][0][1:])])]["genome"]))
                else:  # self mutate
                    parent_genomes.append((self.coords_to_genome[key]["genome"], True))
                if len(parent_genomes) == n_samples:
                    break
        return parent_genomes

    def get_fittest_genome(self):
        """ return fittest genome """
        best = {"fitness": -9999}
        for _, genome in self.coords_to_genome.items():
            if genome["fitness"] >= best["fitness"]:
                best = genome
        return best



def visualise_feature_map(feature_map, dimensions):
    """ visualise n dimensions of the feature map """
    pass
