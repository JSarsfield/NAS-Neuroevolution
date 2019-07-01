"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
import tensorflow as tf
import genes
import cppn


if __name__ == "__main__":
    #pop_size = 2
    #cppn.create_random_graphs(2)
    #cppn = cppn.CPPN(None)
    #print(cppn(np.array([1, 2, 3, 4])))
    gene_pool = genes.GenePool(3)
    print("end")


