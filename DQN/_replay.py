from numpy import zeros
from numpy.random import choice
from numpy import float32, int8

__all__ = ["Replay"]


class Replay:
    def __init__(self, max_size, input_dims):
        self.__mem_size = [max_size, False]
        self.__index = 0

        self.__state_memory = None
        self.__action_memory = None
        self.__reward_memory = None
        self.__terminal_memory = None
        self.__next_state_memory = None

        self.__input_dims = input_dims
        self.__initMemory()

    @property
    def index(self):
        return self.__index

    def exceededMemory(self):
        return self.__mem_size[1]

    def __initMemory(self):
        if isinstance(self.__input_dims, tuple) or isinstance(self.__input_dims, list):
            self.__state_memory = zeros((self.__mem_size[0], *self.__input_dims), dtype=float32)
            self.__next_state_memory = zeros((self.__mem_size[0], *self.__input_dims), dtype=float32)
        else:
            self.__state_memory = zeros((self.__mem_size[0], self.__input_dims), dtype=float32)
            self.__next_state_memory = zeros((self.__mem_size[0], self.__input_dims), dtype=float32)

        self.__action_memory = zeros(self.__mem_size[0], dtype=int8)
        self.__reward_memory = zeros(self.__mem_size[0], dtype=float32)
        self.__terminal_memory = zeros(self.__mem_size[0], dtype=float32)

    def store_transition(self, state, action, reward, next_state, done):
        self.__state_memory[self.__index] = state
        self.__next_state_memory[self.__index] = next_state
        self.__reward_memory[self.__index] = reward
        self.__action_memory[self.__index] = action
        self.__terminal_memory[self.__index] = 1 - done

        self.__index = (self.__index + 1) % self.__mem_size[0]

        if self.__index == 0 and not self.__mem_size[1]:
            self.__mem_size[1] = True

    def sample_buffer(self, batch_size):
        mem_size = min(self.__index, self.__mem_size[0]) if not self.__mem_size[1] else self.__mem_size[0]
        batch = choice(mem_size, batch_size, replace=False)

        states = self.__state_memory[batch]
        next_states = self.__next_state_memory[batch]
        rewards = self.__reward_memory[batch]
        actions = self.__action_memory[batch]
        dones = self.__terminal_memory[batch]

        return states, actions, rewards, next_states, dones
