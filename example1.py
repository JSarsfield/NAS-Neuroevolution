"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from evolution import Evolution


if __name__ == "__main__":
    evolution = Evolution(num_inputs=4)
    evolution.begin_evolution()
    print("end")


