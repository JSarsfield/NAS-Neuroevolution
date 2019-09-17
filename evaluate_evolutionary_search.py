"""
 Evaluate the evolutionary search algorithm and/or compare against another search algorithm
 Calculates metrics such as Global Performance, Global reliability, Precision (opt-in reliability), Coverage
 definitions of which are in paper: Illuminating search spaces by mapping elites. Mouret, Jean-Baptiste; Clune, Jeff
 Visualise performance with graphs of the performance metrics.
 This is required to ensure changes to the evolutionary search algorithm are advantageous.

 __author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

# TODO consider coverage over global performance - how well is it chooses new areas to explore (genome crossover/mutate logic)
# TODO metric for how diverse the behaviours are?
# TODO note we mainly want to illuminate the search space and return genomes that express nets with diverse behaviours
# TODO metric for ranking specialists (elites) and generalists (curiosity)


class EvaluateES:
    """ evaluate evolutionary search algorithms """

    def __init__(self, es_algorithms, es_init_args, evaluation_end_criteria, num_of_runs=1):
        """
        :param es_algorithms: list of evolutionary search algorithms (ES) to evaluate
        :param es_init_args: list of initialisation arguments for each ES
        :param evaluation_end_criteria: criteria to evaluate the ES against
        :param num_of_runs: number of evaluation runs for each ES
        """
        self.es_algorithms = es_algorithms
        self.es_init_args = es_init_args
        self.evaluation_end_criteria = evaluation_end_criteria
        self.num_of_runs = num_of_runs
        self.metrics = {}  # performance metrics for each ES
        self._start_evaluation()

    def _start_evaluation(self):
        for run in self.num_of_runs:
            for i, alg in enumerate(self.es_algorithms):
                self._evaluate_es_algorithm(alg, self.es_init_args[i])

    def _evaluate_es_algorithm(self, alg, args):
        es = alg(args)
        es.begin_evolution()

    def _save_evaluation(self):
        """ save evaluation of the ES algorithms """
        pass


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


