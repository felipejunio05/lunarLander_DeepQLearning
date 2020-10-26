from tensorflow import int8
from tensorflow import zeros
from tensorflow import gather
from tensorflow import float32
from tensorflow import constant
from tensorflow import Variable

from numpy.random import choice

__all__ = ["Replay"]


class Replay:
    def __init__(self, max_size, input_dims):

        self.__mem_size = max_size
        self.__men_states = {"count": 0, "full": False}

        if isinstance(input_dims, tuple) or isinstance(input_dims, list):
            self.__state_memory = Variable(zeros((self.__mem_size, *input_dims), dtype=float32), dtype=float32)
            self.__next_state_memory = Variable(zeros((self.__mem_size, *input_dims), dtype=float32), dtype=float32)
        else:
            self.__state_memory = Variable(zeros((self.__mem_size, input_dims), dtype=float32), dtype=float32)
            self.__next_state_memory = Variable(zeros((self.__mem_size, input_dims), dtype=float32), dtype=float32)

        self.__action_memory = Variable(zeros(self.__mem_size, dtype=int8), dtype=int8)
        self.__reward_memory = Variable(zeros(self.__mem_size, dtype=float32), dtype=float32)
        self.__terminal_memory = Variable(zeros(self.__mem_size, dtype=float32), dtype=float32)

    @property
    def men_states(self):
        return self.__men_states

    def store_transition(self, state, action, reward, next_state, done):
        index = self.__men_states["count"] % self.__mem_size

        self.__state_memory = self.__state_memory[index].assign(state)
        self.__next_state_memory = self.__next_state_memory[index].assign(next_state)
        self.__reward_memory = self.__reward_memory[index].assign(reward)
        self.__action_memory = self.__action_memory[index].assign(action)
        self.__terminal_memory = self.__terminal_memory[index].assign(1 - done)

        if self.__men_states["count"] > self.__mem_size:
            self.__men_states["count"] = 0

            if not self.__men_states["full"]:
                self.__men_states["full"] = True
        else:
            self.__men_states["count"] += 1

    def sample_buffer(self, batch_size):
        max_men = min(self.__men_states["count"], self.__mem_size)

        if max_men > batch_size:
            batch = constant(choice(max_men, batch_size, replace=False))
        else:
            batch = constant(choice(batch_size, batch_size, replace=False))

        states = gather(self.__state_memory, batch)
        next_states = gather(self.__next_state_memory, batch)
        rewards = gather(self.__reward_memory, batch)
        actions = gather(self.__action_memory, batch)
        dones = gather(self.__terminal_memory, batch)

        return states, actions, rewards, next_states, dones
