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
from time import perf_counter
import matplotlib.pyplot as plt
import networkx as nx


class Substrate:
    """ neural network architecture space for finding nodes when expressing a genome to a neural network """

    def __init__(self):
        pass

    def build_network_from_genome(self, genome):
        """" Algorithm 3. express the genome to produce a phenotype (ANN). Return network class. """
        links = []
        nodes = []
        input_nodes = []
        #input_locs = []
        #output_locs = []
        # TODO create input nodes?
        # TODO bias nodes
        # Find input to hidden links and nodes
        for i in np.linspace(-1, 1, genome.num_inputs, dtype=np.float32):
            #input_locs.append((i, float(-1)))
            input_nodes.append(Node(i, float(-1)))
            qtree = QuadTree(genome.graph, var_thresh=genome.var_thresh, band_thresh=genome.band_thresh)
            qtree.division_and_initialisation(i, float(-1))
            new_links = qtree.pruning_and_extraction(i, float(-1))
            links.extend(new_links)
            # Add new nodes
            for link in new_links:
                express_node = True
                for node in nodes:
                    if node.x == link.x2 and node.y == link.y2:
                        express_node = False
                        break
                if express_node:
                    nodes.append(Node(link.x2, link.y2))
        # Find hidden to hidden links and new nodes
        unexplored_nodes = deque()
        unexplored_nodes.extend(nodes)
        # TODO VERY SLOW!!!! optimise this
        print("explore substrate")
        start = perf_counter()
        while unexplored_nodes:
            node = unexplored_nodes.popleft()
            qtree = QuadTree(genome.graph, var_thresh=genome.var_thresh, band_thresh=genome.band_thresh)
            qtree.division_and_initialisation(node.x, node.y)
            new_links = qtree.pruning_and_extraction(node.x, node.y)
            links.extend(new_links)
            # Add only new nodes
            for link in new_links:
                express_node = True
                for e_node in nodes:
                    if e_node.x == link.x2 and e_node.y == link.y2:
                        express_node = False
                        break
                if express_node:
                    new_node = Node(link.x2, link.y2)
                    nodes.append(new_node)
                    unexplored_nodes.append(new_node)
        # Find hidden to output links
        for i in np.linspace(-1, 1, genome.num_outputs, dtype=np.float32):
            #output_locs.append((i, float(1)))
            nodes.append(Node(i, float(1)))
            qtree = QuadTree(genome.graph, var_thresh=genome.var_thresh, band_thresh=genome.band_thresh)
            qtree.division_and_initialisation(i, float(1), outgoing=False)
            new_links = qtree.pruning_and_extraction(i, float(1), outgoing=False)
            links.extend(new_links)
        print("Finished exploring substrate "+str(perf_counter()-start))
        # Remove neurons and their connections that don't have a path from input to output
        # Add link references to relevant nodes
        nodes[0:0] = input_nodes  # extend left, inplace no copying
        for link in links:
            for i, node in enumerate(nodes):
                if node.x == link.x2 and node.y == link.y2:
                    node.ingoing_links.append(link)
                    link.ingoing_node = i
                elif node.x == link.x1 and node.y == link.y1:
                    node.outgoing_links.append(link)
                    link.outgoing_node = i
        # TODO depth first search to find all links on all paths from input to output
        self.depth_first_search(genome, nodes)


        G = nx.DiGraph()
        [G.add_edge((l.x1,l.y1), (l.x2,l.y2)) for l in links]
        paths = list(path for input in input_locs for output in output_locs for path in nx.all_simple_paths(G, source=input, target=output))
        link_locs = list(set([path[node_i] + path[node_i + 1] for path in paths for node_i in range(len(path) - 1)]))
        keep_links = [link for link in links for link_loc in link_locs if link_loc[0] == link.x1 and link_loc[1] == link.y1 and link_loc[2] == link.x2 and link_loc[3] == link.y2]
        print("keep_links "+str(len(keep_links))+ " link_locs "+str(len(link_locs))+" links "+str(len(links)))
        #nx.drawing.nx_pylab.draw(G)
        #plt.show()
        # TODO construct neural network and return it
        # TODO if links is empty initialise empty Network and give lowest score
        return Network(genome, keep_links)

    def depth_first_search(self, genome, nodes):
        """ find links and nodes on paths from input to output nodes """
        keep_links = []
        keep_nodes = []
        # For each input node
        for start in range(genome.num_inputs):
            # for each link in input node
            for out_link in nodes[start].outgoing_links:
                path = deque()
                path.append(out_link)
                while path:
                    if path[-1].y2 == 1:
                        print("")
                    else:



class QuadTree:
    """ Determines hidden node placement within an ANN """

    # TODO evolve/mutate var_thresh and band_threshold - these values passed to children genomes
    def __init__(self, cppn, max_depth=10, var_thresh=0.001, band_thresh=0.001):
        #self.quad_points = []  # Store all QuadPoints in tree
        self.quad_leafs = []  # Quad points that are leaf nodes in the quad tree
        self.cppn = cppn  # Query CPPN graph to get weight of connection
        self.max_depth = max_depth  # The max depth the quadtree will split if variance is still above variance threshold
        self.var_thresh = var_thresh  # When variance of child quads is below this threshold, stop division
        self.band_thresh = band_thresh  # Band threshold for expressing a link if var of neighbours is above this thresh

    def division_and_initialisation(self, a, b, outgoing=True):
        """ Algorithm 1 - a, b represent x1, y1 when outgoing and x2, y2 when ingoing """
        quads_que = deque()  # Contains quads to split x,y,width,level - Add root quad, centre is 0,0
        #self.quad_points.append(QuadPoint(0, 0, 1, 1))
        quads_que.append(QuadPoint(0, 0, 1, 1))
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
                #self.quad_points.extend(q.children)
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
                # TODO Create new link specified by(x1, y1, x2, y2, weight) and scale weight based on weight range(e.g.[-3.0, 3.0])
                if outgoing:
                    if b < q_leaf.y:  # outgoing node b (y) must be less than forward node y
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
        self.level = level  # Number of divisions/iterations from root quad
        self.hw = self.width / 2  # half width
        self.children = []  # Four child quads if not leaf
        self.child_var = None  # Variance of the four child quads
        self.weight = None
