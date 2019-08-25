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
from config import *

# TODO !!! this is still slow when init_substrate_width and init_substrate_height are large


class Substrate:
    """ neural network architecture space for finding nodes when expressing a genome to a neural network """

    def __init__(self):
        pass

    def build_network_from_genome(self, genome, n_net_inputs, n_net_outputs):
        """ Split substrate into rectangles and determine whether a rectangle becomes a node given CPPN """
        # TODO check if node should be expressed given variance with local nodes, width = 2/substrate_width
        nodes = []
        links = []
        input_x_locs = np.linspace(-1, 1, n_net_inputs, dtype=np.float32)
        hidden_x_locs = np.linspace(-1, 1, genome.substrate_width, dtype=np.float32)
        hidden_y_locs = np.linspace(-1, 1, genome.substrate_height+2, dtype=np.float32)[1:-1] # Leave out -1 and 1
        output_x_locs = np.linspace(-1, 1, n_net_outputs, dtype=np.float32)
        neighbour_width = 2/genome.substrate_width
        nodes.append([])
        # Add input nodes
        for i in input_x_locs:
            nodes[-1].append(Node(i, -1))
        # Add suitable hidden nodes
        for layer in hidden_y_locs:
            nodes.append([])
            for node in hidden_x_locs:
                node_weight = genome.graph.forward([node, layer, node, layer])[0].item()
                diff_left = abs(node_weight-genome.graph.forward([node, layer, node - neighbour_width, layer])[0].item())
                diff_right = abs(node_weight-genome.graph.forward([node, layer, node + neighbour_width, layer])[0].item())
                # If min diff is above variance threshold then express
                if max(diff_left, diff_right) > 0.05:  # TODO this needs revisiting, maybe use STEP
                    nodes[-1].append(Node(node, layer))
        nodes.append([])
        # Add output nodes
        for i in output_x_locs:
            nodes[-1].append(Node(i, 1))
        # Add links
        for out_layer in range(len(nodes)-1):
            for out_node in nodes[out_layer]:
                for in_layer in range(out_layer+1, len(nodes)):
                    for in_node in nodes[in_layer]:
                        link_out = genome.graph.forward([out_node.x, out_node.y, in_node.x, in_node.y])
                        weight = link_out[0].item()
                        leo = link_out[1].item()
                        # if express node
                        if leo == 1:
                            link = Link(out_node.x, out_node.y, in_node.x, in_node.y, weight)
                            links.append(link)
                            out_node.add_out_link(link)
                            in_node.add_in_link(link)
                            link.out_node = out_node
                            link.in_node = in_node
        # Depth first search to find all links on all paths from input to output
        keep_links, keep_nodes = self.depth_first_search(nodes[0])
        # Determine if each input and output node is in keep_nodes and thus on path
        if len(keep_nodes) < (n_net_inputs + n_net_outputs) or keep_nodes[n_net_inputs - 1].y != -1 or \
                keep_nodes[-n_net_outputs].y != 1:
            # An input/output node didn't have any outgoing/ingoing links thus neural net is void
            is_void = True
        else:
            is_void = False
        return Network(genome, keep_links, keep_nodes, n_net_inputs, n_net_outputs, void=is_void)

    @staticmethod
    def depth_first_search(input_nodes):
        """ find links and nodes on paths from input to output nodes """
        # TODO rework this to only ensure all output nodes are on a path i.e. dangling input nodes are fine (filtered by evolution)
        path = deque()  # lifo buffer storing currently explored path
        links_2add = deque()  # life buffer storing new links to add if we reach output node
        keep_links = []  # Set of links to keep because they are on a path from input to output
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
                    if len(path[-1]["link"].in_node.outgoing_links) > 0:  # if ingoing node of link also has link then add and keep going forward
                        new_link["link"] = path[-1]["link"].in_node.outgoing_links[0]
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
                        if path[-1]["link"].in_node.y == 1:
                            keep_links.extend([d["link"] for d in links_2add])
                            links_2add.clear()
                        # Node is dangling or output node hit so go back through path
                        is_forward = False
                        continue
                else:
                    # Go back through path until new link then go forward
                    new_ind = path[-1]["ind"]+1
                    if new_ind < len(path[-1]["link"].out_node.outgoing_links):  # If outgoing node of link has more links to explore
                        new_link["link"] = path[-1]["link"].out_node.outgoing_links[new_ind]
                        new_link["ind"] = new_ind
                        if new_link["link"] in keep_links:
                            if len(links_2add) > 0 and links_2add[-1]["link"] == path[-1]["link"]:
                                links_2add.pop()
                            path.pop()
                            continue
                        if len(links_2add) > 0 and links_2add[-1]["link"] == path[-1]["link"]:
                            links_2add.pop()
                        links_2add.append(new_link)
                        path.pop()
                        path.append(new_link)
                        is_forward = True  # new link to explore
                        continue
                    else:  # No unexplored links at this node so keep going back through path
                        if len(links_2add) > 0 and links_2add[-1]["link"] == path[-1]["link"]:
                            links_2add.pop()
                        path.pop()
                        continue
        # Get unique nodes in keep_links
        keep_nodes = []
        for link in keep_links:
            in_node = next((x for x in keep_nodes if x == link.in_node), None)
            if in_node is None:
                keep_nodes.append(link.in_node.copy(link, is_in_node=True))
            else:
                in_node.update_in_node(link)
            out_node = next((x for x in keep_nodes if x == link.out_node), None)
            if out_node is None:
                keep_nodes.append(link.out_node.copy(link, is_in_node=False))
            else:
                out_node.update_out_node(link)
        keep_nodes.sort(key=lambda node: (node.y, node.x))  # Sort nodes by y (layer) then x (pos in layer)
        return keep_links, keep_nodes


class SubstrateESHyperNeat:
    """ neural network architecture space for finding nodes when expressing a genome to a neural network """

    def __init__(self):
        pass

    def build_network_from_genome(self, genome, n_net_inputs, n_net_outputs):
        """" Algorithm 3. express the genome to produce a phenotype (ANN). Return network class. """
        links = []
        nodes = []
        input_nodes = []
        # TODO bias nodes
        start_time = perf_counter()
        # Find input to hidden links and nodes
        for i in np.linspace(-1, 1, n_net_inputs, dtype=np.float32):
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
        #print("input to hid: ", perf_counter()-start_time)
        # Find hidden to hidden links and new nodes
        unexplored_nodes = deque()
        unexplored_nodes.extend(nodes)
        # TODO VERY SLOW!!!! optimise this e.g. use sets for faster time complexity/change approach
        start_time = perf_counter()
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
            # If taking too long give up and return void net
            if perf_counter()-start_time > substrate_search_max_time:
                print("Too long, giving up, net set to void")
                return Network(genome, [], [], n_net_inputs, n_net_outputs, void=True)
        #print("hid to hid: ", perf_counter()-start_time)
        start_time = perf_counter()
        # Find hidden to output links
        for i in np.linspace(-1, 1, n_net_outputs, dtype=np.float32):
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
        #print("hid to output: ", perf_counter()-start_time)
        start_time = perf_counter()
        # Remove neurons and their connections that don't have a path from input to output
        # Add link references to relevant nodes
        nodes[0:0] = input_nodes  # extend left, inplace no copying
        for link in links:
            for i, node in enumerate(nodes):
                if node.x == link.x2 and node.y == link.y2:
                    node.add_in_link(link)
                    link.in_node = node
                elif node.x == link.x1 and node.y == link.y1:
                    node.add_out_link(link)
                    link.out_node = node
        #print("add link refs to nods: ", perf_counter()-start_time)
        start_time = perf_counter()
        # Depth first search to find all links on all paths from input to output
        keep_links, keep_nodes = self.depth_first_search(input_nodes)
        #print("DFS: ", perf_counter()-start_time)
        # Determine that each input and output node is in keep_nodes and thus on path
        if len(keep_nodes) < (n_net_inputs + n_net_outputs) or keep_nodes[n_net_inputs-1].y != -1 or keep_nodes[-n_net_outputs].y != 1:
            # An input/output node didn't have any outgoing/ingoing links thus neural net is void
            is_void = True
        else:
            is_void = False
        return Network(genome, keep_links, keep_nodes, n_net_inputs, n_net_outputs, void=is_void)

    @staticmethod
    def depth_first_search(self, input_nodes):
        """ find links and nodes on paths from input to output nodes """
        # TODO rework this to only ensure all output nodes are on a path i.e. dangling input nodes are fine (filtered by evolution)
        path = deque()  # lifo buffer storing currently explored path
        links_2add = deque()  # life buffer storing new links to add if we reach output node
        keep_links = []  # Set of links to keep because they are on a path from input to output
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
                    if len(path[-1]["link"].in_node.outgoing_links) > 0:  # if ingoing node of link also has link then add and keep going forward
                        new_link["link"] = path[-1]["link"].in_node.outgoing_links[0]
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
                        if path[-1]["link"].in_node.y == 1:
                            keep_links.extend([d["link"] for d in links_2add])
                            links_2add.clear()
                        # Node is dangling or output node hit so go back through path
                        is_forward = False
                        continue
                else:
                    # Go back through path until new link then go forward
                    new_ind = path[-1]["ind"]+1
                    if new_ind < len(path[-1]["link"].out_node.outgoing_links):  # If outgoing node of link has more links to explore
                        new_link["link"] = path[-1]["link"].out_node.outgoing_links[new_ind]
                        new_link["ind"] = new_ind
                        if new_link["link"] in keep_links:
                            if len(links_2add) > 0 and links_2add[-1]["link"] == path[-1]["link"]:
                                links_2add.pop()
                            path.pop()
                            continue
                        if len(links_2add) > 0 and links_2add[-1]["link"] == path[-1]["link"]:
                            links_2add.pop()
                        links_2add.append(new_link)
                        path.pop()
                        path.append(new_link)
                        is_forward = True  # new link to explore
                        continue
                    else:  # No unexplored links at this node so keep going back through path
                        if len(links_2add) > 0 and links_2add[-1]["link"] == path[-1]["link"]:
                            links_2add.pop()
                        path.pop()
                        continue
        # Get unique nodes in keep_links
        keep_nodes = []
        for link in keep_links:
            in_node = next((x for x in keep_nodes if x == link.in_node), None)
            if in_node is None:
                keep_nodes.append(link.in_node.copy(link, is_in_node=True))
            else:
                in_node.update_in_node(link)
            out_node = next((x for x in keep_nodes if x == link.out_node), None)
            if out_node is None:
                keep_nodes.append(link.out_node.copy(link, is_in_node=False))
            else:
                out_node.update_out_node(link)
        keep_nodes.sort(key=lambda node: (node.y, node.x))  # Sort nodes by y (layer) then x (pos in layer)
        return keep_links, keep_nodes


class QuadTree:
    """ Determines hidden node placement within an ANN """

    # TODO evolve/mutate var_thresh and band_threshold - these values passed to children genomes
    def __init__(self, cppn, var_thresh=0.001, band_thresh=0.001):
        self.quad_leafs = []  # Quad points that are leaf nodes in the quad tree
        self.cppn = cppn  # Query CPPN graph to get weight of connection
        self.var_thresh = var_thresh  # When variance of child quads is below this threshold, stop division
        self.band_thresh = band_thresh  # Band threshold for expressing a link if var of neighbours is above this thresh

    def division_and_initialisation(self, a, b, outgoing=True):
        """ Algorithm 1 - a, b represent x1, y1 when outgoing and x2, y2 when ingoing """
        quads_que = deque()  # Contains quads to split x,y,width,level - Add root quad, centre is 0,0
        quads_que.append(QuadPoint(0, 0, 1, 1))
        out = self.cppn.forward([0, 1, 0, 1])
        quads_que[-1].weight = out[0].item()
        quads_que[-1].leo = out[0].item()
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
                    out = self.cppn.forward([a, b, child.x, child.y])
                    child.weight = out[0].item()
                    child.leo = out[1].item()
                else:
                    out = self.cppn.forward([child.x, child.y, a, b])
                    child.weight = out[0].item()
                    child.leo = out[1].item()
                child_weights = np.append(child_weights, child.weight)
            q.child_var = child_weights.var()
            # Divide until initial resolution or if variance is still high
            if q.level == 1 or (q.level < quad_tree_max_depth and q.child_var > self.var_thresh):
                quads_que.extend(q.children)
                if q.level != 1 and q.leo > 0:
                    self.quad_leafs.append(q)
            else:
                if q.leo > 0:  # LEO must be greater than zero for potential expression
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
            # Express connection if neighbour variance is above band threshold
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
        self.leo = None  # Express if greater than zero
