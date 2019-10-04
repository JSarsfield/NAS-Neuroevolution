"""
Environment to evaluate a neural network during its lifetime

__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""

import numpy as np
from random import randrange
from time import sleep
from feature_dimensions import PerformanceDimension, PhenotypicDimension


def get_env_spaces(gym_env_string):
    """ Get environment observation and action space for gym reinforcement environments """
    import gym
    temp_env = gym.make(gym_env_string)
    return temp_env.observation_space.shape[0], 1 if "Discrete" in str(type(temp_env.action_space)) else temp_env.action_space.shape[0]


class Environment:
    """ base class for all environments """

    def __init__(self, feature_dims=None):
        self.feature_dims = feature_dims
        self.setup_feature_dimensions()

    def setup_feature_dimensions(self):
        self.performance_dims = []
        self.phenotypic_dims = []
        for dim in self.feature_dims:
            if isinstance(dim, PerformanceDimension):
                self.performance_dims.append(dim)
            elif isinstance(dim, PhenotypicDimension):
                self.phenotypic_dims.append(dim)

    def calc_performance_dims(self, *args):
        for dim in self.performance_dims:
            dim.call(*args)

    def calc_phenotypic_dims(self, *args):
        for dim in self.phenotypic_dims:
            dim.call(*args)


class EnvironmentReinforcement(Environment):
    """ Reinforcement environments """

    def __init__(self, gym_env_string, parallel=True, trials=1, steps=900, feature_dims=None):
        super().__init__(feature_dims)
        self.net = None  # Neural network to evaluate
        self.trials = trials  # Fitness = average of all trials
        self.steps = steps  # How many steps should
        self.gym_env_string = gym_env_string
        self.env = None
        self.parallel = parallel

    def evaluate(self, net, render=False):
        """ evaluate the neural net and return the final fitness """
        if net.is_void:
            return -9999  # return low fitness for void networks
        if render:
            import keyboard
        import gym
        self.net = net
        self.env = gym.make(self.gym_env_string)
        fitness = np.array([])
        for trial in range(self.trials if render is False else 999):
            observation = self.env.reset()
            action = self.net.graph(observation.astype(np.float32))  # self.net.graph.forward(observation).max(0)[1].item()
            trial_reward = 0
            for step in range(self.steps):
                if render:
                    if keyboard.is_pressed('q'):
                        self.env.close()
                        return
                    elif keyboard.is_pressed('r'):
                        break
                    else:
                        self.env.render()
                #action = int(action[0]) if len(action) == 1 else action
                observation, reward, done, info = self.env.step(action)
                trial_reward += reward
                if done:
                    break
                action = self.net.graph(observation.astype(np.float32)) # self.net.graph.forward(observation).max(0)[1].item()
            fitness = np.append(fitness, trial_reward)
        if render:
            try:
                self.env.close()
            except:
                print("FAILED to close env during render. Class EnvironmentReinforcement Def evaluate")
        self.net.set_fitness(fitness.max())
        self.calc_performance_dims(self.net)
        self.calc_phenotypic_dims(self.net)
        net.genome.performance_dims = self.performance_dims
        net.genome.phenotypic_dims = self.phenotypic_dims
        return fitness.max()


class EnvironmentReinforcementCustom(Environment):
    """ Reinforcement environments custom (non gym env) """

    def __init__(self, env_class, trials=10000):
        super().__init__()
        self.net = None  # Neural network to evaluate
        self.opponents = []  # List of opponent nets
        self.trials = trials  # Fitness = average of all trials
        self.env = env_class(trials)
        self.num_eval_rounds = trials

    def evaluate(self, nets, render=False):
        """ evaluate the neural net and return the final fitness """
        from game import Game
        game = Game(10000)
        game.start_game(1, [2,3,4,5,6])
        while game.total_rounds_so_far < self.num_eval_rounds:
            for action in game.game_loop():
                action = int(np.random.choice([0, 1, 2], 1, p=[0.5, 0.15, 0.35])[0])
                if action == 0:  # check/call
                    game._bot_check_call()
                elif action == 1:  # bet/raise/all-in
                    bet_max = game.bots[game.bot_to_act].stack + game.bots[game.bot_to_act].bet
                    bet_min = game.current_bet + game.raise_min
                    if bet_max <= bet_min:  # all-in
                        game._bot_bet(bet_max)
                    else:
                        game._bot_bet(randrange(bet_min, bet_max))
                else:
                    game._bot_fold()
            game.new_game()
            print("new game")
        print("all rounds evaluated")
        return 0


class EnvironmentClassification(Environment):
    """ Classification environments (supervised learning with labelled dataset (lifetime learning)) """
    # TODO !!! be careful of overfitting during evolution consider creating a probability distribution of the dataset and sampling from that
    # TODO also reset weights before training and evaluating on new test data

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

