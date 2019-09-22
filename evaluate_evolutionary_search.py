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
        os.mkdir(self.save_dir)

    def run_evaluation(self):
        """ start evaluating the ES algorithms """
        print("start of evaluation")
        for i, _ in enumerate(self.es_algorithms):
            self._setup_algorithm(i)
            self.metrics.append({"index": i, "runs": []})
            for run in range(self.num_of_runs):
                self.metrics[-1]["runs"].append({"gens": []})
                self._evaluate_es_algorithm(self.es_init_args)
                self._save_evaluation()
        os.system("git checkout master")
        print("end of evaluation")

    def _setup_algorithm(self, i):
        """ setup algorithm: checkout branch, compile """
        global evolution
        if "master" in self.es_algorithms[i]:
            os.system("git -c user.name=Joe -c user.email=joe.sarsfield@gmail.com stash --all")
            os.system("git checkout "+self.es_algorithms[i])
            import evolution
        else:
            os.system("git checkout tags/" + self.es_algorithms[i])
            import imp, sys
            for module in sys.modules.values():
                imp.reload(module)
        #os.system("python setup.py sdist")
        print("")

    def _evaluate_es_algorithm(self, args):
        """ evaluate a single ES algorithm for one run """
        es = evolution.Evolution(**args, evaluator_callback=self.generation_complete_callback)
        es.begin_evolution()

    def _save_evaluation(self):
        """ save evaluation of the ES algorithm """
        pass

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
        # TODO global visualisation imports here
        self._load_evaluation()
        self._visualise()

    def _load_evaluation(self):
        """ load evaluation from file """
        pass

    def _visualise(self):
        """ visualise the performance metrics of an evaluation """
        pass


