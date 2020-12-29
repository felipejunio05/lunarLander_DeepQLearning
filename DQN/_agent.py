from numpy import mean
from numpy import arange
from numpy.random import choice
from numpy.random import random

from os.path import exists

from ._replay import Replay
from ._neuralnetwork import NeuralNetwork

__all__ = ["Agent"]


class Agent:
    def __init__(self, lr, gamma, epsilon, epsilon_min, epsilon_decay, n_actions, input_dims, batch_size, mem_size, saveIn='dqn_model.h5'):
        self.__actionSpace = arange(n_actions)
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.__epsMin = epsilon_min
        self.__epsDec = epsilon_decay
        self.__batchSize = batch_size
        self.__saveIn = saveIn
        self.__memory = Replay(mem_size, input_dims)
        self.__q_eval = self.__build_dqn(lr, n_actions, input_dims, 32, 32, 32)

    @property
    def gamma(self):
        return self.__gamma

    @property
    def epsilon(self):
        return self.__epsilon

    @property
    def epsMin(self):
        return self.__epsMin

    @property
    def epsDec(self):
        return self.__epsDec

    @property
    def batchSize(self):
        return self.__batchSize

    @property
    def modelFile(self):
        return self.__saveIn

    @property
    def memory(self):
        return self.__memory

    @staticmethod
    def __build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims, fc3_dims):
        return NeuralNetwork(input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, lr)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state, training=True):
        if training and random() < self.epsilon:
            action = choice(self.__actionSpace)
        else:
            actions = self.__q_eval.predict(state.reshape(1, -1))
            action = actions.argmax()

        return action

    def learn(self):
        if self.__memory.index > self.__batchSize or self.__memory.exceededMemory():
            states, actions, rewards, next_states, dones = self.__memory.sample_buffer(self.__batchSize)

            q_eval = self.__q_eval.predict(states, True)
            q_next = self.__q_eval.predict(next_states, True)

            batch_index = arange(self.__batchSize, dtype="int32")
            q_eval[batch_index, actions] = rewards + self.gamma * q_next.max(axis=1) * dones

            self.__q_eval.train(states, q_eval)

            if (self.epsilon - self.__epsDec) > self.__epsMin:
                self.__epsilon = self.epsilon - self.__epsDec
            else:
                if not self.__epsilon == self.__epsMin:
                    self.__epsilon = self.__epsMin

    def save_model(self):
        self.__q_eval.save(self.__saveIn)

    def load_model(self):
        self.__q_eval.load_model(self.__saveIn)

    def run(self, episodes, env, view=None, graph=None, load_model=None):
        avg_score = []
        epsilons = []

        if load_model is not None:
            if exists(load_model):
                self.__q_eval.load_model(load_model)

        if view is not None:
            view.deiconify()

            if graph is not None:
                view.frameUpdater3(graph.plot(None, None, None, True))

        for episode in range(episodes):
            state = env.reset()

            scores = []
            done = False
            score = 0

            while not done:
                action = self.choose_action(state)
                new_state, reward, done, info = env.step(action)

                if view is not None:
                    view.gymRender = env.render(mode="rgb_array")
                    view.agentActivation = self.__q_eval.getActivation()

                self.store_transition(state, action, reward, new_state, done)
                self.learn()

                score += reward
                scores.append(score)

                state = new_state

            if graph is not None:
                avg_score.append(mean(scores))
                epsilons.append(self.__epsilon)

                imgGraph = graph.plot(episode, epsilons, avg_score)

            if view is not None:
                view.episode(str(episode + 1).zfill(2))

                if graph is not None:
                    view.frameUpdater3(imgGraph)
            else:
                print(f'Episode: {episode} Score: {mean(scores):.2f} Epsilon: {self.__epsilon:.2f}')

        self.save_model()

        if graph is not None:
            graph.save()

        if view is not None:
            view.close()
