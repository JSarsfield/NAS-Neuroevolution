"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from environment import EnvironmentReinforcement, EnvironmentReinforcementCustom
from evaluate_evolutionary_search import EvaluateES
from config import Exec
from game import Game


def single_run():
    from evolution import Evolution
    evolution = Evolution(pop_size=64,
                          environment_type=EnvironmentReinforcement,
                          env_name="BipedalWalker-v2",
                          # CartPole-v0 BipedalWalker-v2 MountainCarContinuous-v0 HandManipulateBlock-v0
                          worker_list=None,  # "hpc_worker_list" "hpc_worker_list_home"
                          session_name=None,  # if None new evolutionary search will be started
                          gen=6)
    evolution.begin_evolution()


def evaluation():
    args = {"pop_size": 32,
            "environment_type": EnvironmentReinforcement,
            "env_args": ["BipedalWalker-v2"],
            "session_name": None,
            "gen": None,
            "execute": Exec.PARALLEL_LOCAL,
            "worker_list": "hpc_worker_list_home",
            "persist_every_n_gens": -1}
    evaluator = EvaluateES(es_algorithms=["master", "v0.3"],
                           es_init_args=args,
                           num_of_runs=1,
                           stop_criterion=2)
    evaluator.run_evaluation()


if __name__ == "__main__":
    print("begin")
    if __debug__:
        print("running unoptimised, consider using -O flag")
    else:
        print("OPTIMISED")
    #single_run()
    evaluation()
    print("end")
