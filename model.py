"""
    trained model

    __author__ = "Joe Sarsfield"
    __email__ = "joe.sarsfield@gmail.com"
"""
from substrate import Substrate
import numpy as np


class NN_bag_model:
    """ neural network bagging model """

    def __init__(self, inputs, outputs):
        self.performance_gen = []  # list of performance per generation
        self.inputs = inputs
        self.outputs = outputs

    def predict(self, genomes, samples, labels):
        """ make a prediction of a set of samples and return performance metric """
        networks = []
        for genome in genomes:
            genome.create_graph()
            networks.append(Substrate().build_network_from_genome(genome, self.inputs, self.outputs))
            networks[-1].init_graph()
        self.tp = 0
        self.fn = 0
        self.tn = 0
        self.fp = 0
        for i, sample in enumerate(samples):
            y_norms = []
            # Get predictions from all nets
            for net in networks:
                y, y_norm, y_arg, y_arg_true = net.predict(sample[-1], labels[i])
                y_norms.append(y_norm)
            y = [0, 1] if np.array(y_norms)[:, 1].sum() >= len(y_norms)/2 else [1, 0]
            y_arg = np.argmax(y)
            y_arg_true = np.argmax(labels[i])
            if y_arg == y_arg_true:  # TP or TN
                if y_arg == 1:
                    self.tp += 1
                else:
                    self.tn += 1
            else:  # FP or FN
                if y_arg == 0:  # FN
                    self.fn += 1
                else:  # FP
                    self.fp += 1
        self.visualise_performance()
        self.performance_gen.append(networks[0].auc(self.tp, self.tn, self.fp, self.fn))
        for genome in genomes:
            genome.net = None
            genome.graph = None
        return self.performance_gen[-1]

    def visualise_performance(self):
        pass

    def save_model(self):
        """ save model to disk """
        pass

    def load_model(self):
        """ load model from disk """
        pass