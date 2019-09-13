"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from evolution import Evolution
from environment import EnvironmentReinforcement


if __name__ == "__main__":
    evolution = Evolution(n_net_inputs=4,
                          n_net_outputs=2,
                          pop_size=512,
                          environment=EnvironmentReinforcement,
                          gym_env_string="BipedalWalker-v2",  # CartPole-v0 BipedalWalker-v2
                          worker_list="hpc_worker_list",  # "hpc_worker_list"
                          processes=64)
    evolution.begin_evolution()
    print("end")
