import tensorflow as tf
from numpy import arange
from numpy.random import choice
from numpy.random import random


__all__ = ["Agent"]


class Replay:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.men_states = {"count": 0, "full": False}

        self.state_memory = tf.Variable(tf.zeros((self.mem_size, *input_dims), dtype=tf.float32), dtype=tf.float32)
        self.next_state_memory = tf.Variable(tf.zeros((self.mem_size, *input_dims), dtype=tf.float32), dtype=tf.float32)
        self.action_memory = tf.Variable(tf.zeros(self.mem_size, dtype=tf.int8), dtype=tf.int8)
        self.reward_memory = tf.Variable(tf.zeros(self.mem_size, dtype=tf.float32), dtype=tf.float32)
        self.terminal_memory = tf.Variable(tf.zeros(self.mem_size, dtype=tf.float32), dtype=tf.float32)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.men_states["count"] % self.mem_size

        self.state_memory = self.state_memory[index].assign(state)
        self.next_state_memory = self.next_state_memory[index].assign(next_state)
        self.reward_memory = self.reward_memory[index].assign(reward)
        self.action_memory = self.action_memory[index].assign(action)
        self.terminal_memory = self.terminal_memory[index].assign(1 - done)

        if self.men_states["count"] > self.mem_size:
            self.men_states["count"] = 0

            if not self.men_states["full"]:
                self.men_states["full"] = True
        else:
            self.men_states["count"] += 1

    def sample_buffer(self, batch_size):
        max_men = min(self.men_states["count"], self.mem_size)

        if max_men > batch_size:
            batch = tf.constant(choice(max_men, batch_size, replace=False))
        else:
            batch = tf.constant(choice(batch_size, batch_size, replace=False))

        states = tf.gather(self.state_memory, batch)
        next_states = tf.gather(self.next_state_memory, batch)
        rewards = tf.gather(self.reward_memory, batch)
        actions = tf.gather(self.action_memory, batch)
        dones = tf.gather(self.terminal_memory, batch)

        return states, actions, rewards, next_states, dones


class Agent:

    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=1e-2, mem_size=10 ** 6, fname='dqn_model_01.h5'):

        self.action_space = arange(n_actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.eps_dec = epsilon_dec
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = Replay(mem_size, input_dims)
        self.q_eval = self.build_dqn(lr, n_actions, *input_dims, 32, 32, 32)

    def build_dqn(self, lr, n_actions, input_dims, fc1_dims, fc2_dims, fc3_dims):
        return NeuralNetwork(input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, lr)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        if random() < self.epsilon:
            action = choice(self.action_space)
        else:
            actions = self.q_eval.predict(state.reshape(1, -1))
            action = actions.argmax()

        return action

    def learn(self):
        if self.memory.men_states["count"] > self.batch_size or self.memory.men_states["full"]:
            states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

            q_eval = self.q_eval.predict(states, True)
            q_next = self.q_eval.predict(next_states, True)

            batch_index = arange(self.batch_size, dtype="int32")
            q_eval[batch_index, actions] = rewards + self.gamma * q_next.max(axis=1) * dones

            self.q_eval.train(states, q_eval)
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval.load_model(self.model_file)


class NeuralNetwork:
    def __init__(self, n_states, h1_layer, h2_layer, h3_layer, n_actions, lr):

        self.optimizer = tf.optimizers.Adam(lr)
        self.activation = [[], [], [], []]

        self.weights = {'F_1': [tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=(n_states, h1_layer)), name="F_1/W"),
                                tf.Variable(tf.zeros(shape=(h1_layer, )), name="F_1/B")],

                        'F_2': [tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=(h1_layer, h2_layer)), name="F_2/W"),
                                 tf.Variable(tf.zeros(shape=(h2_layer, )), name="F_2/B")],

                        'F_3': [tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=(h2_layer, h3_layer)), name="F_3/W"),
                                tf.Variable(tf.zeros(shape=(h3_layer,)), name="F_3/B")],

                        'F_4': [tf.Variable(tf.random.uniform(minval=-1, maxval=1, shape=(h3_layer, n_actions)), name="F_4/W"),
                                tf.Variable(tf.zeros(shape=(n_actions, )), name="F_4/B")]}

        with open("Config/Model.json") as config:
            self.model_base = tf.keras.models.model_from_json(config.read())

    def fullyConneted(self, data, train=False):

        with tf.name_scope("FC_1") as scope:
            H_01_INPUT = tf.matmul(data, self.weights['F_1'][0])
            H_01_OUTPUT = tf.nn.relu(H_01_INPUT + self.weights['F_1'][1], name="FC_01")

        with tf.name_scope("FC_2") as scope:
            H_02_INPUT = tf.matmul(H_01_OUTPUT, self.weights['F_2'][0])
            H_02_OUTPUT = tf.nn.relu(H_02_INPUT + self.weights['F_2'][1], name="FC_02")

        with tf.name_scope("FC_3") as scope:
            H_03_INPUT = tf.matmul(H_02_OUTPUT, self.weights['F_3'][0])
            H_03_OUTPUT = tf.nn.relu(H_03_INPUT + self.weights['F_3'][1], name="FC_03")

        with tf.name_scope("FC_4") as scope:
            Y_PRED = tf.matmul(H_03_OUTPUT, self.weights['F_4'][0]) + self.weights['F_4'][1]

        if not train:
            self.activation[0] = H_01_OUTPUT[0].numpy()
            self.activation[1] = [H_02_OUTPUT[0].numpy(), (tf.transpose(H_01_OUTPUT) * self.weights['F_2'][0]).numpy()]
            self.activation[2] = [H_03_OUTPUT[0].numpy(), (tf.transpose(H_02_OUTPUT) * self.weights['F_3'][0]).numpy()]
            self.activation[3] = [Y_PRED[0].numpy(), (tf.transpose(H_03_OUTPUT) * self.weights['F_4'][0]).numpy()]

        return Y_PRED

    def loss(self, y_true, y_pred):
        return tf.losses.mean_squared_error(y_true, y_pred)

    def train(self, x_value, y_value):
        weights = self.get_weights()

        with tf.GradientTape() as tape:
            tape.watch(weights)

            y_pred = self.fullyConneted(x_value, train=True)
            f_loss = self.loss(y_value, y_pred)

        grads = tape.gradient(f_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

    def predict(self, x_value, learning=False):
        if learning:
            y_pred = self.fullyConneted(x_value, train=True).numpy()
        else:
            y_pred = self.fullyConneted(x_value, train=False).numpy()

        return y_pred

    def get_weights(self):
        weights = []

        for key in self.weights.keys():
            weights.extend(self.weights[key])

        return weights

    def getActivation(self):
        return self.activation

    def load_model(self, file):
        self.model_base = tf.keras.models.load_model(file)

        for i, k in zip(range(len(self.model_base.layers)), self.weights.keys()):
            w, b = self.model_base.layers[i].get_weights()
            self.weights[k] = [tf.Variable(w, name=f"F_{i}/W"), tf.Variable(b, name=f"F_{i}/B")]

    def save(self, file):
        for i, k in zip(range(len(self.model_base.layers)), self.weights.keys()):
            self.model_base.layers[i].set_weights([self.weights[k][0].numpy(), self.weights[k][1].numpy()])

        self.model_base.save(file)
