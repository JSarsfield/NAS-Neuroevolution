"""
Environment to evaluate a neural network during its lifetime
"""

import numpy as np


class Environment:
    """ base class for all environments """

    def __init__(self):
        pass


class EnvironmentReinforcement(Environment):
    """ Reinforcement environments """

    def __init__(self, gym_env_string="CartPole-v0", trials=10, steps=1000):
        super().__init__()
        self.net = None  # Neural network to evaluate
        self.trials = trials  # Fitness = average of all trials
        self.steps = steps  # How many steps should
        self.gym_env_string = gym_env_string
        self.env = None

    def evaluate(self, net, render=False):
        import gym
        self.net = net
        self.env = gym.make(self.gym_env_string)
        fitness = np.array([])
        """ evaluate the neural net and return the final fitness """
        for trial in range(self.trials):
            observation = self.env.reset()
            action = self.net.graph.forward(observation).max(0)[1].item()
            trial_reward = 0
            for step in range(self.steps):
                if render:
                    self.env.render()
                observation, reward, done, info = self.env.step(action)
                trial_reward += reward
                if done:
                    break
                action = self.net.graph.forward(observation).max(0)[1].item()
            fitness = np.append(fitness, trial_reward)
        self.net.set_fitness(fitness.mean())
        print("fitness", self.net.fitness)


class EnvironmentClassification(Environment):
    """ Classification environments (supervised learning with labelled dataset) """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

