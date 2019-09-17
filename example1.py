"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from evolution import Evolution
from environment import EnvironmentReinforcement


if __name__ == "__main__":
    evolution = Evolution(pop_size=64,
                          environment_type=EnvironmentReinforcement,
                          env_name="BipedalWalker-v2",  # CartPole-v0 BipedalWalker-v2 MountainCarContinuous-v0 HandManipulateBlock-v0
                          worker_list=None,  # "hpc_worker_list" "hpc_worker_list_home"
                          session_name=None,  # if None new evolutionary search will be started
                          gen=6)
    evolution.begin_evolution()
    print("end")
