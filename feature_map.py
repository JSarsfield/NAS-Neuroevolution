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
            for dim in genome.genomic_dims:
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
                    res = tree.query([key], crossover_neighbour_elites)
                    neighbours = [res[1][0][i] for i in range(1, len(res[1][0])) if res[0][0][i] != np.inf]
                    parent_genomes.append((self.coords_to_genome[key]["genome"],
                                           self.coords_to_genome[tuple(tree.data[random.choice(neighbours)])]["genome"]))
                else:  # self mutate
                    parent_genomes.append((self.coords_to_genome[key]["genome"], True))
                if len(parent_genomes) == n_samples:
                    break
        return parent_genomes

    def get_fittest_genomes(self, n=1):
        """ return n fittest genome. Fetch multiple networks when bagging. """
        return sorted([item[1]['genome'] for item in self.coords_to_genome.items()], key=lambda x: x.fitness)[-n:]

    def visualise(self):
        """ visualise n dimensions of the feature map """
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib import pyplot as plt
        import mpl_toolkits.mplot3d.axes3d as p3
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = []
        y = []
        z = []
        for key in self.coords_to_genome.keys():
            x.append(key[0])
            y.append(key[1])
            z.append(self.coords_to_genome[key]["fitness"])
        ax.scatter(x, y, z)
        ax.set_xlabel('Num nodes genome')
        ax.set_ylabel('Sum link weights')
        ax.set_zlabel('Fitness')
        plt.show()
        print("")
