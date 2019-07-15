""" Substrate - contains the neural network architecture space whereby a node and connection can be expressed

Logic for expressing/decoding a genome into a network (ANN)/ phenotype


Paper: An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density and Connectivity of Neurons
See algorithm 1 and 2 in appendices

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
from collections import deque  # Faster than using list
from network import Network, Link, Node


class Substrate:
    """ neural network architecture space for finding nodes when expressing a genome to a neural network """

    def __init__(self):
        pass

    def build_network_from_genome(self, genome):
        """" Algorithm 3. express the genome to produce a phenotype (ANN). Return network class. """
        net = Network(genome)
        links = []
        nodes = []
        # TODO create input nodes?
        # Find input to hidden links
        for i in np.linspace(-1, 1, genome.num_inputs, dtype=np.float32):
            qtree = QuadTree(genome.graph)
            qtree.division_and_initialisation(i, float(-1))
            new_links = qtree.pruning_and_extraction(i, float(-1))
            links.extend(new_links)
            # TODO express the nodes given the newly expressed links
            for link in new_links:
                express_node = True
                for node in nodes:
                    if node.x == link.x2 and node.y == link.y2:
                        express_node = False
                        break
                if express_node:
                    nodes.append(Node(link.x2, link.y2))
        # Find hidden to hidden links
        

        # Find hidden to output links
        for i in range(genome.num_outputs):
            pass


class QuadTree:
    """ Determines hidden node placement within an ANN """

    def __init__(self, cppn, max_depth=10, var_thresh=0.001, band_thresh=0.001):
        self.quad_points = []  # Store all QuadPoints in tree
        self.quad_leafs = []  # Quad points that are leaf nodes in the quad tree
        self.cppn = cppn  # Query CPPN graph to get weight of connection
        self.max_depth = max_depth  # The max depth the quadtree will split if variance is still above variance threshold
        self.var_thresh = var_thresh  # When variance of child quads is below this threshold, stop division
        self.band_thresh = band_thresh  # Band threshold for expressing a link if var of neighbours is above this thresh

    def division_and_initialisation(self, a, b, outgoing=True):
        """ Algorithm 1 - a, b represent x1, y1 when outgoing and x2, y2 when ingoing """
        quads_que = deque()  # Contains quads to split x,y,width,level - Add root quad, centre is 0,0
        self.quad_points.append(QuadPoint(0, 0, 1, 1))
        quads_que.append(self.quad_points[-1])
        # While quads is not empty continue dividing
        while quads_que:
            q = quads_que.popleft()
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
                    child.weight = self.cppn.query(child.x, child.y, a, b)[0]
                child_weights = np.append(child_weights, child.weight)
            q.child_var = child_weights.var()
            # Divide until initial resolution or if variance is still high
            if q.level == 1 or (q.level < self.max_depth and q.child_var > self.var_thresh):
                self.quad_points.extend(q.children)
                quads_que.extend(q.children)
            else:
                q.is_leaf = True
                self.quad_leafs.append(q)

    def pruning_and_extraction(self, a, b, outgoing=True):
        """ Algorithm 2 - a, b represent x1, y1 when outgoing and x2, y2 when ingoing """
        links = []  # Store expressed links
        # For each quad leaf
        for q_leaf in self.quad_leafs:
            # Determine if point is in a band by checking neighbour CPPN values
            if outgoing:
                dif_left = abs(q_leaf.weight - self.cppn.query(a, b, q_leaf.x - q_leaf.width, q_leaf.y)[0])
                dif_right = abs(q_leaf.weight - self.cppn.query(a, b, q_leaf.x + q_leaf.width, q_leaf.y)[0])
                dif_bottom = abs(q_leaf.weight - self.cppn.query(a, b, q_leaf.x, q_leaf.y - q_leaf.width)[0])
                dif_top = abs(q_leaf.weight - self.cppn.query(a, b, q_leaf.x, q_leaf.y + q_leaf.width)[0])
            else:
                dif_left = abs(q_leaf.weight - self.cppn.query(q_leaf.x - q_leaf.width, q_leaf.y, a, b)[0])
                dif_right = abs(q_leaf.weight - self.cppn.query(q_leaf.x + q_leaf.width, q_leaf.y, a, b)[0])
                dif_bottom = abs(q_leaf.weight - self.cppn.query(q_leaf.x, q_leaf.y - q_leaf.width, a, b)[0])
                dif_top = abs(q_leaf.weight - self.cppn.query(q_leaf.x, q_leaf.y + q_leaf.width, a, b)[0])
            # Express connection if neighbour variance if above band threshold
            if  max(min(dif_left, dif_right), min(dif_bottom, dif_top)) > self.band_thresh:
                # Create new link specified by(x1, y1, x2, y2, weight) and scale weight based on weight range(e.g.[-3.0, 3.0])
                if outgoing:
                    links.append(Link(a, b, q_leaf.x, q_leaf.y, q_leaf.weight))
                else:
                    links.append(Link(q_leaf.x, q_leaf.y, a, b, q_leaf.weight))
        return links  # Return expressed links


class QuadPoint:
    """ quad point in the quadtree x, y, width, level """

    def __init__(self, x, y, width, level):
        self.is_leaf = False
        self.x = x  # x centre
        self.y = y  # y centre
        self.width = width  # width of quad. Width is term used in paper but is also length of side
        self.level = level  # Number of divisions from root quad
        self.hw = self.width / 2  # half width
        self.children = []  # Four child quads if not leaf
        self.child_var = None  # Variance of the four child quads
        self.weight = None
