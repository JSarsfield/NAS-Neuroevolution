"""
Speciate the genomes to encourage and protect diversity allowing new avenues of problem search space to be explored

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from config import *
import numpy as np

class Species:

    def __init__(self, representative_genome):
        self.age = 0  # Number of generations this species has existed
        self.representative_genome = representative_genome  # The genome that this species represents when calculating distance to new genomes
        self.genomes = [representative_genome]

    def get_distance(self, new_genome):
        """ get distance between new_genome and representative_genome to determine if belongs to this species"""
        disjoint = 0
        weight_diff = np.array([])
        i = 0
        j = 0
        max_genes = max(len(new_genome.gene_links), len(self.representative_genome.gene_links))
        while i < len(new_genome.gene_links) and j < len(self.representative_genome.gene_links):
            # If same gene calculate weight diff - disabled link = weight of 0
            if new_genome.gene_links[i].historical_marker == self.representative_genome.gene_links[j].historical_marker:
                l1 = new_genome.gene_links[i].weight if new_genome.gene_links[i].enabled else 0
                l2 = self.representative_genome.gene_links[j].weight if self.representative_genome.gene_links[j].enabled else 0
                weight_diff = np.append(weight_diff, abs(l1-l2))
                i += 1
                j += 1
            else:  # Different gene so increment disjoint
                disjoint += 1
                if new_genome.gene_links[i].historical_marker < self.representative_genome.gene_links[j].historical_marker:
                    i += 1
                else:
                    j += 1
        # Add excess genes if any
        excess = max(len(new_genome.gene_links)-i, len(self.representative_genome.gene_links)-j)
        return ((compatibility_excess_coeff*excess)/max_genes)+((compatibility_disjoint_coeff*disjoint)/max_genes)+(compatibility_weight_coeff*weight_diff.mean())

    def add_to_species(self, new_genome):
        self.genomes.append(new_genome)