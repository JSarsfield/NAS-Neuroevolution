from genes import GeneNode
from genome import CPPNGenome
from substrate import Substrate
from config import *
import numpy as np
from activations import ActivationFunctionSet, NodeFunctionSet
import ray
import os


@ray.remote(num_cpus=1)
def parallel_reproduce_eval(parents, n_net_inputs, n_net_outputs, env, env_args):
    if __debug__:  # TODO delete
        print("running unoptimised, consider using -O flag")
    else:
        print("OPTIMISED")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    results = []
    for parent in parents:
        # Reproduce from parent genomes
        if type(parent[-1]) is not bool:  # Two parent genomes so crossover
            genome, new_structures = crossover(parent[0], parent[1])
        else:  # One parent genome so mutate
            genome, new_structures = copy_with_mutation(parent[0], mutate=parent[1])
        # Create genome graph
        genome.create_graph()
        # Create net from genome
        net = Substrate().build_network_from_genome(genome, n_net_inputs, n_net_outputs)
        if not net.is_void:
            net.init_graph()  # init TF graph
            # Evaluate
            fitness = env(*env_args).evaluate(net)
            net.set_fitness(fitness)  # Note this also sets fitness in genome
            net.clear_sessions()  # cleanup
            net.graph = None  # delete TF graph
        if __debug__:
            if net.is_void:
                print("void net returned")
            else:
                print(str(fitness) + " returned")
        genome.net = None
        net = None
        genome.graph = None  # delete pytorch graph
        results.append((genome, new_structures))
    return results


def crossover(g1, g2):
    """ crossover of two parent genomes """
    i = 0
    j = 0
    nodes_to_add = []
    links_to_add = []
    sub_width = g1.substrate_width if g1.fitness >= g2.fitness else g2.substrate_width
    sub_height = g1.substrate_height if g1.fitness >= g2.fitness else g2.substrate_height
    while i < len(g1.gene_links) or j < len(g2.gene_links):
        if i < len(g1.gene_links) and j < len(g2.gene_links):
            if g1.gene_links[i].historical_marker == g2.gene_links[j].historical_marker:
                if g1.fitness >= g2.fitness:
                    links_to_add.append(g1.gene_links[i])
                    nodes_to_add.append(g1.gene_links[i].in_node)
                    nodes_to_add.append(g1.gene_links[i].out_node)
                else:
                    links_to_add.append(g2.gene_links[j])
                    nodes_to_add.append(g2.gene_links[j].in_node)
                    nodes_to_add.append(g2.gene_links[j].out_node)
                i += 1
                j += 1
            elif g1.gene_links[i].historical_marker < g2.gene_links[j].historical_marker:
                nodes_to_add.append(g1.gene_links[i].in_node)
                nodes_to_add.append(g1.gene_links[i].out_node)
                links_to_add.append(g1.gene_links[i])
                i += 1
            else:
                nodes_to_add.append(g2.gene_links[j].in_node)
                nodes_to_add.append(g2.gene_links[j].out_node)
                links_to_add.append(g2.gene_links[j])
                j += 1
        else:
            if j == len(g2.gene_links):
                nodes_to_add.append(g1.gene_links[i].in_node)
                nodes_to_add.append(g1.gene_links[i].out_node)
                links_to_add.append(g1.gene_links[i])
                i += 1
            else:
                nodes_to_add.append(g2.gene_links[j].in_node)
                nodes_to_add.append(g2.gene_links[j].out_node)
                links_to_add.append(g2.gene_links[j])
                j += 1
    return create_new_genome(nodes_to_add, links_to_add, g1.cppn_inputs, sub_width, sub_height)


def copy_with_mutation(g1, mutate=True):
    """ copy a genome with mutation """
    return create_new_genome(g1.gene_nodes_in + g1.gene_nodes,
                             g1.gene_links,
                             g1.cppn_inputs,
                             g1.substrate_width,
                             g1.substrate_height,
                             mutate=mutate)


def create_new_genome(nodes_to_add, links_to_add, cppn_inputs, sub_width, sub_height, mutate=True):
    """ Create new genome - perform structural & non structural mutation """
    gene_nodes = set()
    gene_links = []
    # Add in/out nodes and links
    for node in nodes_to_add:
        gene_nodes.add(GeneNode(node.depth,
                                node.act_func,
                                node.node_func,
                                node.historical_marker,
                                can_modify=node.can_modify,
                                enabled=node.enabled,
                                bias=node.bias))
    for link in links_to_add:
        gene_links.append((link.weight,
                           link.in_node.historical_marker,
                           link.out_node.historical_marker,
                           link.historical_marker,
                           link.enabled))
    gene_nodes = list(gene_nodes)
    gene_nodes.sort(key=lambda x: x.depth)
    if mutate:
        new_structures = mutate_structural(gene_nodes, gene_links)  # This is performed on master thread to ensure only new genes are added to gene pool
    else:
        new_structures = []
    gene_nodes_in = gene_nodes[:cppn_inputs]
    gene_nodes = gene_nodes[cppn_inputs:]
    new_genome = CPPNGenome(gene_nodes_in, gene_nodes, gene_links, substrate_width=sub_width, substrate_height=sub_height)
    if mutate:
        new_genome.mutate_nonstructural()
    return new_genome, new_structures


def get_node_from_hist_marker(gene_nodes, hist_marker):
    for node in gene_nodes:
        if node.historical_marker == hist_marker:
            return node
    raise Exception("No node with historical marker found in func get_node_from_hist_marker evolution.py")


def mutate_structural(gene_nodes, gene_links):
    """ mutate genome to add nodes and links. Note passed by reference """
    # TODO !!! allow nodes to become disabled and thus all ingoing out going links disabled
    # TODO add config probs for toggling NODE! enable/disable
    new_structures = {}  # keep track of new links and nodes to ensure correct historical marker is applied after
    # Mutate attempt add link
    if event(link_add_prob):
        # shuffle node indices
        inds = np.random.choice(len(gene_nodes), len(gene_nodes), replace=False)
        attempt = 0
        gene_links.sort(key=lambda x: x[1])
        for in_node_ind in inds:
            if gene_nodes[in_node_ind].depth == 0:
                continue
            else:
                ind_node_before = None
                for i in range(in_node_ind, 0, -1):
                    if gene_nodes[i].depth != gene_nodes[in_node_ind].depth:
                        ind_node_before = i
                        break
                # check if node has a potential new link
                existing_links = []
                for i, link in enumerate(gene_links):
                    if link[1] == gene_nodes[in_node_ind].historical_marker:
                        existing_links.append(link[2])
                existing_links = np.unique(existing_links)
                # If number of existing ingoing links is less than the number of nodes before this node then add a new link
                if len(existing_links) != ind_node_before+1:
                    existing_links = [i for i, node in enumerate(gene_nodes) if node.historical_marker in existing_links]
                    out_node = gene_nodes[np.random.choice(np.setdiff1d(np.arange(ind_node_before+1), existing_links), 1)[0]]
                    # Add new link
                    #new_link = gene_pool.get_or_create_gene_link(gene_nodes[in_node_ind].historical_marker, out_node.historical_marker)
                    gene_links.append((random.uniform(weight_init_min, weight_init_max),
                                           gene_nodes[in_node_ind].historical_marker,
                                           out_node.historical_marker,
                                           None,
                                           True))
                    new_structures["new_link"] = (gene_links[-1][1], gene_links[-1][2])
                    break
                attempt += 1
            if attempt == new_link_attempts:
                break
    # Mutate add node with random activation function
    if event(node_add_prob):
        # Get a random link to split and add node
        gene_link_ind = random.randint(0, len(gene_links)-1)
        old_link = gene_links[gene_link_ind]
        in_node = get_node_from_hist_marker(gene_nodes, old_link[1])
        out_node = get_node_from_hist_marker(gene_nodes, old_link[2])
        old_link_update = (old_link[0], old_link[1], old_link[2], old_link[3], False)
        del gene_links[gene_link_ind]
        gene_links.append(old_link_update)
        act_set = ActivationFunctionSet()
        node_set = NodeFunctionSet()
        # Create new node
        new_structures["new_node"] = (old_link[1], old_link[2])
        new_structures["node_depth"] = out_node.depth+((in_node.depth-out_node.depth)/2)
        gene_nodes.append(GeneNode(new_structures["node_depth"],
                                   act_set.get_random_activation_func(),
                                   node_set.get("dot"),
                                   None,
                                   True,
                                   True,
                                   random.uniform(bias_init_min, bias_init_max)))
        # Create new link going into new node
        gene_links.append((1,
                           in_node.historical_marker,
                           gene_nodes[-1].historical_marker,
                           None,
                           True))
        # Create new link going out of new node
        gene_links.append((old_link[0],
                           gene_nodes[-1].historical_marker,
                           out_node.historical_marker,
                           None,
                           True))
    gene_nodes.sort(key=lambda x: x.depth)
    return new_structures
