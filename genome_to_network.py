"""

Logic for expressing/decoding a genome into a network (ANN)/ phenotype


Paper: An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density and Connectivity of Neurons
See algorithm 1 and 2 in appendices

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
from collections import deque  # Faster than using list


class Substrate:
    """ neural network architecture space for finding nodes when expressing a genome to a neural network """

    def __init__(self):
        pass


class QuadTree:
    """ Determines hidden node placement within an ANN """

    def __init__(self, cppn):
        self.quad_points = []  # Store all QuadPoints in tree TODO determine if we only need to store leaf nodes, if so append if no further split required
        self.cppn = cppn  # CPPN genome to produce Quad Tree from

    def division_and_initialisation(self, a, b, outgoing=True):
        """ Algorithm 1 """
        quads = deque(QuadPoint(0,0,1,1))   # Contains quads to split x,y,width,level - Add root quad, centre is 0,0
        # While quads is not empty continue dividing
        while quads:
            q = quads.popleft()
            q.is_leaf = False
            qlev1 = q.level + 1
            q.children.append(QuadPoint(q.x - q.hw, q.y - q.hw, q.hw, qlev1))
            q.children.append(QuadPoint(q.x - q.hw, q.y + q.hw, q.hw, qlev1))
            q.children.append(QuadPoint(q.x + q.hw, q.y + q.hw, q.hw, qlev1))
            q.children.append(QuadPoint(q.x + q.hw, q.y - q.hw, q.hw, qlev1))
            self.quad_points.append(q)
            self.quad_points.extend(q.children)
            for child in q.children:
                if outgoing:
                    child.weight = self.cppn.graph.query(a, b, child.x, child.y)[0]
                else:
                    child.weight = self.cppn.graph.query(child.x, child, y, a, b)[0]
            # TODO Divide until initial resolution or if variance is still high

    def pruning_and_extraction(self):
        """ Algorithm 2 """
        pass


class QuadPoint:
    """ quad point in the quadtree x, y, width, level """

    def __init__(self, x, y, width, level):
        self.is_leaf = True
        self.x = x  # x centre
        self.y = y # y centre
        self.width = width
        self.level = level # depth
        self.hw = self.width / 2  # half width
        self.children = []  # Four child quads if not leaf
        self.weight = None

    def __iter__(self):
        return self
