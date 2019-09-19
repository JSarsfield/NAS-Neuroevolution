"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from evolution import Evolution
from environment import EnvironmentReinforcement
from evaluate_evolutionary_search import EvaluateES
from config import Exec


def single_run():
    evolution = Evolution(pop_size=64,
                          environment_type=EnvironmentReinforcement,
                          env_name="BipedalWalker-v2",
                          # CartPole-v0 BipedalWalker-v2 MountainCarContinuous-v0 HandManipulateBlock-v0
                          worker_list=None,  # "hpc_worker_list" "hpc_worker_list_home"
                          session_name=None,  # if None new evolutionary search will be started
                          gen=6)
    evolution.begin_evolution()


def evaluation():
    args = {"pop_size": 512,
            "environment_type": EnvironmentReinforcement,
            "env_name": "BipedalWalker-v2",
            "session_name": None,
            "gen": None,
            "execute": Exec.PARALLEL_HPC,
            "worker_list": "hpc_worker_list",
            "persist_every_n_gens": -1}
    evaluator = EvaluateES(es_algorithms=[Evolution],
                           es_init_args=[args],
                           num_of_runs=1,
                           stop_criterion=40)
    evaluator.run_evaluation()


if __name__ == "__main__":
    print("begin")
    #single_run()
    evaluation()
    print("end")
