"""
    Feature/behaviour map as defined in map-elites.

    __author__ = "Joe Sarsfield"
    __email__ = "joe.sarsfield@gmail.com"
"""


class FeatureMap:
    """ feature map containing dimensions of interest, including fitness metric/s and feature/behaviour metrics """

    def __init__(self, load_map=None):
        self.map = []

    def dimension_of_interest(self):
        """ feature/behaviour of interest, multiple dimensions can exist within the map """
        pass

    def performance_dimension(self):
        """ every map has atleat one performance/fitness metric dimension """
        pass

    def update_feature_map(self, genomes):
        """ updates cells of feature map if genome performance is highest in cell or cell empty """
        pass

    def sample_feature_map(self, n_samples):
        """ sample n genomes from random cells in the feature map """
        # TODO consider dynamic bias for selecting cells for crossover
        pass


def visualise_feature_map(feature_map, dimensions):
    """ visualise n dimensions of the feature map """
    pass
