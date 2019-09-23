"""
 Evaluate the evolutionary search algorithm and/or compare against another search algorithm
 Calculates metrics such as Global Performance, Global reliability, Precision (opt-in reliability), Coverage
 definitions of which are in paper: Illuminating search spaces by mapping elites. Mouret, Jean-Baptiste; Clune, Jeff
 Visualise performance with graphs of the performance metrics.
 This is required to ensure changes to the evolutionary search algorithm are advantageous.

 __author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import pickle
import os
import datetime
import importlib
from evolution import Evolution

# TODO consider coverage over global performance - how well is it chooses new areas to explore
#  (genome crossover/mutate logic)
# TODO metric for how diverse the behaviours are?
# TODO note we mainly want to illuminate the search space and return genomes that express nets with diverse behaviours
# TODO metric for ranking specialists (elites) and generalists (curiosity)


class EvaluateES:
    """ evaluate evolutionary search algorithms """

    def __init__(self,
                 es_algorithms,
                 es_init_args,
                 num_of_runs=1,
                 stop_criterion=10):
        """
        :param es_algorithms: list of evolutionary search algorithms (ES) to evaluate
        :param es_init_args: list of initialisation arguments for each ES
        :param num_of_runs: number of evaluation runs for each ES
        :param stop_criterion: criterion to stop evaluation of a single run e.g. time, generation, global performance
        """
        # TODO load previous version/s of our ES algorithm from git commit (tag each version)
        self.es_algorithms = es_algorithms
        self.es_init_args = es_init_args
        self.stop_criterion = stop_criterion
        self.num_of_runs = num_of_runs
        self.metrics = []  # performance metrics for each ES
        self.evaluation_name = "evaluation_" + str(datetime.datetime.now()).replace(" ", "_")
        self.save_dir = "./evaluations/" + self.evaluation_name + "/"
        try:
            os.mkdir(self.save_dir)
        except:
            os.mkdir("./evaluations/")
            os.mkdir(self.save_dir)

    def run_evaluation(self):
        """ start evaluating the ES algorithms """
        print("start of evaluation")
        self.metrics.append({"index": 0, "runs": []})
        for run in range(self.num_of_runs):
            self.metrics[-1]["runs"].append({"gens": []})
            self._evaluate_es_algorithm(self.es_init_args)
            self._save_evaluation()
        print("end of evaluation")

    def _evaluate_es_algorithm(self, args):
        """ evaluate a single ES algorithm for one run """
        es = Evolution(**args, evaluator_callback=self.generation_complete_callback)
        es.begin_evolution()

    def _save_evaluation(self):
        """ save evaluation of the ES algorithm """
        with open(self.save_dir + "eval.pkl", "wb") as f:
            pickle.dump(self.metrics, f)

    def generation_complete_callback(self, current_gen, generation_best_fitness):
        """ callback for when an ES algorithm finishes evaluating a generation of genomes """
        # get generation metrics
        self.metrics[-1]["runs"][-1]["gens"].append(self._get_metrics(generation_best_fitness))
        self._get_metrics(generation_best_fitness)
        return self._stop_criterion_gen_reached(current_gen)

    def _stop_criterion_gen_reached(self, current_gen):
        """ generation reached evaluation stop criterion for a single run """
        if current_gen == self.stop_criterion:
            return True
        return False

    def _get_metrics(self, generation_best_fitness):
        """ get metrics of ES algorithm """
        # TODO measure time between generations!!!
        return {"gen_glob_perf": generation_best_fitness, "gen_glob_reli": 0, "precision": 0, "coverage": 0}


class VisualiseEvaluation:
    """ visualise the evaluation of ES algorithms """

    def __init__(self, eval_file):
        global plt, np
        import matplotlib.pyplot as plt
        import numpy as np
        self._load_evaluation(eval_file)
        self._visualise()

    def _load_evaluation(self, eval_file):
        """ load evaluation from file """
        with open(eval_file, "rb") as f:
            self.metrics = pickle.load(f)

    def _visualise(self):
        """ visualise the performance metrics of an evaluation """
        gen_glob_perfs = []
        for i, run in enumerate(self.metrics[-1]["runs"]):
            gen_glob_perfs.append([])
            for gen in run["gens"]:
                gen_glob_perfs[-1].append(gen["gen_glob_perf"])
        gen_glob_perfs = np.array(gen_glob_perfs)
        plt.errorbar(list(range(1, gen_glob_perfs.shape[-1]+1)), gen_glob_perfs.mean(axis=0), gen_glob_perfs.std(axis=0), linestyle='None', marker='^')
        plt.show()
        print("")


