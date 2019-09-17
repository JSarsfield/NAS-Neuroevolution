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

# TODO consider coverage over global performance - how well is it chooses new areas to explore
#  (genome crossover/mutate logic)
# TODO metric for how diverse the behaviours are?
# TODO note we mainly want to illuminate the search space and return genomes that express nets with diverse behaviours
# TODO metric for ranking specialists (elites) and generalists (curiosity)


class EvaluateES:
    """ evaluate evolutionary search algorithms """

    def __init__(self, es_algorithms, es_init_args, evaluation_end_criteria, num_of_runs=1):
        """
        :param es_algorithms: list of evolutionary search algorithms (ES) to evaluate
        :param es_init_args: list of initialisation arguments for each ES
        :param evaluation_end_criteria: criteria to evaluate the ES against e.g. time, generation, global performance
        :param num_of_runs: number of evaluation runs for each ES
        """
        # TODO load previous version/s of our ES algorithm from git commit (tag each version)
        self.es_algorithms = es_algorithms
        self.es_init_args = es_init_args
        self.evaluation_end_criteria = evaluation_end_criteria
        self.num_of_runs = num_of_runs
        self.metrics = []  # performance metrics for each ES
        self.evaluation_name = "evaluation_" + str(datetime.datetime.now()).replace(" ", "_")
        self.save_dir = "./evaluations/" + self.evaluation_name + "/"
        os.mkdir(self.save_dir)

    def run_evaluation(self):
        """ start evaluating the ES algorithms """
        print("start of evaluation")
        for i, alg in enumerate(self.es_algorithms):
            self.metrics.append({"index": i, "runs": []})
            for run in range(self.num_of_runs):
                self.metrics[-1]["runs"].append({"gens": []})
                self._evaluate_es_algorithm(alg, self.es_init_args[i])
                self._save_evaluation()
        print("end of evaluation")

    def _evaluate_es_algorithm(self, alg, args):
        """ evaluate a single ES algorithm for one run """
        es = alg(**args, evaluator_callback=self.generation_complete_callback)
        es.begin_evolution()

    def _save_evaluation(self):
        """ save evaluation of the ES algorithm """
        pass

    def generation_complete_callback(self, generation_best_fitness):
        """ callback for when an ES algorithm finishes evaluating a generation of genomes """
        # get generation metrics
        self.metrics[-1]["runs"][-1]["gens"].append(self._get_metrics())
        # TODO check if evaluation end criteria is met
        return True

    def _get_metrics(self):
        """ get metrics of ES algorithm """
        return {"glob_perf": 0, "glob_reli": 0, "precision": 0, "coverage": 0}


class VisualiseEvaluation:
    """ visualise the evaluation of ES algorithms """

    def __init__(self, eval_file):
        # TODO global visualisation imports here
        self._load_evaluation()
        self._visualise()

    def _load_evaluation(self):
        """ load evaluation from file """
        pass

    def _visualise(self):
        """ visualise the performance metrics of an evaluation """
        pass


