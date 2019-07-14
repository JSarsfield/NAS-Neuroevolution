""" Substrate - contains the neural network architecture space whereby a node and connection can be expressed

Logic for expressing/decoding a genome into a network (ANN)/ phenotype


Paper: An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density and Connectivity of Neurons
See algorithm 1 and 2 in appendices

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
from collections import deque  # Faster than using list
from network import Network


class Substrate:
    """ neural network architecture space for finding nodes when expressing a genome to a neural network """

    def __init__(self):
        pass

    def build_network_from_genome(self, genome):
        """" express the genome to produce a phenotype (ANN). Return network class. """
        net = Network(genome)
        # Find input to hidden links
        for i in range(genome.num_inputs):
            pass
        # Find hidden to hidden links

        # Find hidden to output links
        for i in range(genome.num_outputs):
            pass


class QuadTree:
    """ Determines hidden node placement within an ANN """

    def __init__(self, cppn, max_depth=10, var_thresh=0.001):
        self.quad_points = []  # Store all QuadPoints in tree TODO determine if we only need to store leaf nodes, if so append if no further split required
        self.cppn = cppn  # CPPN genome to produce Quad Tree from
        self.max_depth = max_depth  # The max depth the quadtree will split if variance is still above variance threshold
        self.var_thresh = var_thresh  # When variance of child quads is below this threshold, stop division

    def division_and_initialisation(self, a, b, outgoing=True):
        """ Algorithm 1 """
        quads = deque(QuadPoint(0,0,1,1))   # Contains quads to split x,y,width,level - Add root quad, centre is 0,0
        self.quad_points.append(quads[0])
        # While quads is not empty continue dividing
        while quads:
            q = quads.popleft()
            qlev1 = q.level + 1
            q.children.append(QuadPoint(q.x - q.hw, q.y - q.hw, q.hw, qlev1))
            q.children.append(QuadPoint(q.x - q.hw, q.y + q.hw, q.hw, qlev1))
            q.children.append(QuadPoint(q.x + q.hw, q.y + q.hw, q.hw, qlev1))
            q.children.append(QuadPoint(q.x + q.hw, q.y - q.hw, q.hw, qlev1))
            child_weights = np.array([])
            for child in q.children:
                if outgoing:
                    child.weight = self.cppn.query(a, b, child.x, child.y)[0]
                else:
                    child.weight = self.cppn.query(child.x, child, y, a, b)[0]
                child_weights = np.append(child_weights, child.weight)
            q.child_var = child_weights.var()
            # TODO Divide until initial resolution or if variance is still high
            if q.level == 1 or (q.level < self.max_depth and q.child_var > self.var_thresh):
                self.quad_points.extend(q.children)
                q.is_leaf = False
                quads.deque(q.children)
            print("")



    def pruning_and_extraction(self):
        """ Algorithm 2 """
        pass


class QuadPoint:
    """ quad point in the quadtree x, y, width, level """

    def __init__(self, x, y, width, level):
        self.is_leaf = True
        self.x = x  # x centre
        self.y = y  # y centre
        self.width = width
        self.level = level  # depth
        self.hw = self.width / 2  # half width
        self.children = []  # Four child quads if not leaf
        self.child_var = None  # Variance of the four child quads given
        self.weight = None

    def __iter__(self):
        return self
