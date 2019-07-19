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
from itertools import chain


class Substrate:
    """ neural network architecture space for finding nodes when expressing a genome to a neural network """

    def __init__(self):
        pass

    def build_network_from_genome(self, genome, n_net_inputs, n_net_outputs):
        """" Algorithm 3. express the genome to produce a phenotype (ANN). Return network class. """
        links = []
        nodes = []
        input_nodes = []
        # TODO bias nodes
        # Find input to hidden links and nodes
        for i in np.linspace(-1, 1, genome.num_inputs, dtype=np.float32):
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
            nodes.append(Node(i, float(1)))
            qtree = QuadTree(genome.graph, var_thresh=genome.var_thresh, band_thresh=genome.band_thresh)
            qtree.division_and_initialisation(i, float(1), outgoing=False)
            new_links = qtree.pruning_and_extraction(i, float(1), outgoing=False)
            # Only keep new links where the outgoing hidden node exists
            new_links_keep = []
            for n_link in new_links:
                for node in nodes:
                    if node.x == n_link.x1 and node.y == n_link.y1:
                        new_links_keep.append(n_link)
                        break
            links.extend(new_links_keep)
        print("Finished exploring substrate "+str(perf_counter()-start))
        # Remove neurons and their connections that don't have a path from input to output
        # Add link references to relevant nodes
        nodes[0:0] = input_nodes  # extend left, inplace no copying
        for link in links:
            for i, node in enumerate(nodes):
                if node.x == link.x2 and node.y == link.y2:
                    node.ingoing_links.append(link)
                    link.ingoing_node = node
                elif node.x == link.x1 and node.y == link.y1:
                    node.outgoing_links.append(link)
                    link.outgoing_node = node
        # Depth first search to find all links on all paths from input to output
        keep_links, keep_nodes = self.depth_first_search(input_nodes)
        if len(keep_nodes) <= n_net_outputs or keep_nodes[-n_net_outputs].y != 1:
            # An input/output node didn't have any outgoing/ingoing links thus neural net is void
            is_void = True
            print("neural network is void")
        else:
            is_void = False
        return Network(genome, keep_links, keep_nodes, n_net_inputs, n_net_outputs, void=is_void)

    def depth_first_search(self, input_nodes):
        """ find links and nodes on paths from input to output nodes """
        path = deque()  # lifo buffer storing currently explored path
        links_2add = deque()  # life buffer storing new links to add if we reach output node
        keep_links = []  # List of links to keep because they are on a path from input to output
        for input_ind, input in enumerate(input_nodes):
            # Each element is a dict with link reference and local index of outgoing node's
            if len(input.outgoing_links) == 0:
                return [], []  # An input neuron has no links and thus this neural network is void
            path.append({"link": input.outgoing_links[0], "ind": 0})
            links_2add.append(path[-1])
            is_forward = True  # False when link to dangling node
            # while unexplored links on from this input node exist
            while path:
                new_link = {}
                if is_forward:
                    if len(path[-1]["link"].ingoing_node.outgoing_links) > 0:  # if ingoing node of link also has link then add and keep going forward
                        new_link["link"] = path[-1]["link"].ingoing_node.outgoing_links[0]
                        new_link["ind"] = 0
                        if new_link["link"] in keep_links:  # If we reach a link on the keep_links path then add links_2add and go back
                            keep_links.extend([d["link"] for d in links_2add])
                            links_2add.clear()
                            is_forward = False
                            continue
                        path.append(new_link)
                        links_2add.append(new_link)
                    else:  # No new links to explore
                        # Check if node is output
                        if path[-1]["link"].ingoing_node.y == 1:
                            keep_links.extend([d["link"] for d in links_2add])
                            links_2add.clear()
                        # Node is dangling or output node hit so go back through path
                        is_forward = False
                        continue
                else:
                    # Go back through path until new link then go forward
                    new_ind = path[-1]["ind"]+1
                    if new_ind < len(path[-1]["link"].outgoing_node.outgoing_links):  # If outgoing node of link has more links to explore
                        new_link["link"] = path[-1]["link"].outgoing_node.outgoing_links[new_ind]
                        new_link["ind"] = new_ind
                        is_forward = True  # new link to explore
                        if len(links_2add) > 0 and path[-1]["link"] == links_2add[-1]["link"]:
                            links_2add.pop()
                        links_2add.append(new_link)
                        path.pop()
                        path.append(new_link)
                        continue
                    else:  # No unexplored links at this node so keep going back through path
                        path.pop()
                        continue
        # Get unique nodes in keep_links
        keep_nodes = list(set(chain.from_iterable((link.ingoing_node, link.outgoing_node) for link in keep_links)))
        keep_nodes.sort(key=lambda node: (node.y, node.x))  # Sort nodes by y (layer) then x (pos in layer)
        return keep_links, keep_nodes


class QuadTree:
    """ Determines hidden node placement within an ANN """

    # TODO evolve/mutate var_thresh and band_threshold - these values passed to children genomes
    def __init__(self, cppn, max_depth=10, var_thresh=0.001, band_thresh=0.001):
        self.quad_leafs = []  # Quad points that are leaf nodes in the quad tree
        self.cppn = cppn  # Query CPPN graph to get weight of connection
        self.max_depth = max_depth  # The max depth the quadtree will split if variance is still above variance threshold
        self.var_thresh = var_thresh  # When variance of child quads is below this threshold, stop division
        self.band_thresh = band_thresh  # Band threshold for expressing a link if var of neighbours is above this thresh

    def division_and_initialisation(self, a, b, outgoing=True):
        """ Algorithm 1 - a, b represent x1, y1 when outgoing and x2, y2 when ingoing """
        quads_que = deque()  # Contains quads to split x,y,width,level - Add root quad, centre is 0,0
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
                    child.weight = self.cppn.forward([a, b, child.x, child.y])[0].item()
                else:
                    child.weight = self.cppn.forward([child.x, child.y, a, b])[0].item()
                child_weights = np.append(child_weights, child.weight)
            q.child_var = child_weights.var()
            # Divide until initial resolution or if variance is still high
            if q.level == 1 or (q.level < self.max_depth and q.child_var > self.var_thresh):
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
                dif_left = abs(q_leaf.weight - self.cppn.forward([a, b, q_leaf.x - q_leaf.width, q_leaf.y])[0].item())
                dif_right = abs(q_leaf.weight - self.cppn.forward([a, b, q_leaf.x + q_leaf.width, q_leaf.y])[0].item())
                dif_bottom = abs(q_leaf.weight - self.cppn.forward([a, b, q_leaf.x, q_leaf.y - q_leaf.width])[0].item())
                dif_top = abs(q_leaf.weight - self.cppn.forward([a, b, q_leaf.x, q_leaf.y + q_leaf.width])[0].item())
            else:
                dif_left = abs(q_leaf.weight - self.cppn.forward([q_leaf.x - q_leaf.width, q_leaf.y, a, b])[0].item())
                dif_right = abs(q_leaf.weight - self.cppn.forward([q_leaf.x + q_leaf.width, q_leaf.y, a, b])[0].item())
                dif_bottom = abs(q_leaf.weight - self.cppn.forward([q_leaf.x, q_leaf.y - q_leaf.width, a, b])[0].item())
                dif_top = abs(q_leaf.weight - self.cppn.forward([q_leaf.x, q_leaf.y + q_leaf.width, a, b])[0].item())
            # Express connection if neighbour variance if above band threshold
            if max(min(dif_left, dif_right), min(dif_bottom, dif_top)) > self.band_thresh:
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
