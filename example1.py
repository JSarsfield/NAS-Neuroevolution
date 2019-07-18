"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from evolution import Evolution


if __name__ == "__main__":
    evolution = Evolution(n_net_inputs=4, n_net_outputs=1, pop_size=10000)
    evolution.begin_evolution()
    print("end")
