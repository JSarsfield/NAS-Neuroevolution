"""
Example for running neuroevolution code

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
from environment import EnvironmentClassification, EnvironmentReinforcement, EnvironmentReinforcementCustom
from evaluate_evolutionary_search import EvaluateES, VisualiseEvaluation
from config import Exec
#from game import Game
import feature_dimensions


def single_run():
    from evolution import Evolution
    evolution = Evolution(pop_size=128,
                          environment_type=EnvironmentClassification,
                          env_args=['Oakfield_data.csv', 2], # CartPole-v0 BipedalWalker-v2 MountainCarContinuous-v0 HandManipulateBlock-v0
                          session_name=None,  # if None new evolutionary search will be started
                          gen=None,
                          execute=Exec.PARALLEL_LOCAL,
                          worker_list="hpc_worker_list_home",
                          persist_every_n_gens=-1,
                          log_to_driver=False)
    evolution.begin_evolution()


def evaluation():
    args = {"pop_size": 256,
            "environment_type": EnvironmentReinforcement,
            "env_args": ['BipedalWalker-v2'],
            "session_name": None,
            "gen": None,
            "execute": Exec.PARALLEL_LOCAL,
            "worker_list": "hpc_worker_list_home",
            "persist_every_n_gens": -1,
            "log_to_driver": False}
    evaluator = EvaluateES(es_algorithms=[],
                           es_init_args=args,
                           num_of_runs=4,
                           stop_criterion=1000)
    evaluator.run_evaluation()


def start():
    print("begin")
    if __debug__:
        print("running unoptimised, consider using -O flag")
    else:
        print("OPTIMISED")
    # single_run()
    evaluation()
    # VisualiseEvaluation("./evaluations/evaluation_2019-09-22_22:48:46.407820/eval.pkl")
    print("end")


if __name__ == "__main__":
    print("begin")
    if __debug__:
        print("running unoptimised, consider using -O flag")
    else:
        print("OPTIMISED")
    single_run()
    # TODO save/load feature map
    # TODO Load top x performing nets from feature map for classification/regression bagging model
    #evaluation()
    # VisualiseEvaluation("./evaluations/evaluation_2019-10-07_13:53:56.919307/eval.pkl")
    print("end")
