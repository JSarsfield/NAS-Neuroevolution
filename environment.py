"""
Environment to evaluate a neural network during its lifetime
"""

import numpy as np


def get_env_spaces(gym_env_string):
    """ Get environment observation and action space for gym reinforcement environments """
    import gym
    temp_env = gym.make(gym_env_string)
    return temp_env.observation_space.shape[0], 1 if "Discrete" in str(type(temp_env.action_space)) else temp_env.action_space.shape[0]

"""
def parallel_evaluate_net(net, env, gym_env_string):
   
    return env(gym_env_string).evaluate(net)
"""


class Environment:
    """ base class for all environments """

    def __init__(self):
        pass


class EnvironmentReinforcement(Environment):
    """ Reinforcement environments """

    def __init__(self, gym_env_string, parallel=True, trials=2, steps=10000):
        super().__init__()
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
        for trial in range(self.trials):
            observation = self.env.reset()
            action = self.net.graph(observation.astype(np.float32)).numpy()  # self.net.graph.forward(observation).max(0)[1].item()
            trial_reward = 0
            for step in range(self.steps):
                if render:
                    if keyboard.is_pressed('q'):
                        self.env.close()
                        return
                    else:
                        self.env.render()
                action = int(action[0]) if len(action) == 1 else action
                observation, reward, done, info = self.env.step(action)
                trial_reward += reward
                if done:
                    break
                action = self.net.graph(observation.astype(np.float32)).numpy()  # self.net.graph.forward(observation).max(0)[1].item()
            fitness = np.append(fitness, trial_reward)
        if render:
            try:
                self.env.close()
            except:
                pass
        self.net.set_fitness(fitness.max())
        print("fitness", self.net.fitness)
        return fitness.max()


class EnvironmentClassification(Environment):
    """ Classification environments (supervised learning with labelled dataset (lifetime learning)) """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

