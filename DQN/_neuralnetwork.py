from tensorflow import nn
from tensorflow import math
from tensorflow import zeros
from tensorflow import random
from tensorflow import matmul
from tensorflow import losses
from tensorflow import float32
from tensorflow import function
from tensorflow import constant
from tensorflow import Variable
from tensorflow import transpose
from tensorflow import TensorSpec
from tensorflow import name_scope
from tensorflow import optimizers
from tensorflow import GradientTape
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

__all__ = ["NeuralNetwork"]


class NeuralNetwork:
    def __init__(self, n_states, h1_layer, h2_layer, h3_layer, n_actions, lr):
        self.__optimizer = optimizers.Adam(lr)
        self.__activation = []

        self.__weights = {'F_1': [Variable(random.uniform(minval=-math.sqrt(6/n_states), maxval=math.sqrt(6/n_states), shape=(n_states, h1_layer)), name="F_1/W"),
                                  Variable(zeros(shape=(h1_layer, )), name="F_1/B")],

                          'F_2': [Variable(random.uniform(minval=-math.sqrt(6/h1_layer), maxval=math.sqrt(6/h1_layer), shape=(h1_layer, h2_layer)), name="F_2/W"),
                                  Variable(zeros(shape=(h2_layer, )), name="F_2/B")],

                          'F_3': [Variable(random.uniform(minval=-math.sqrt(6/h2_layer), maxval=math.sqrt(6/h2_layer), shape=(h2_layer, h3_layer)), name="F_3/W"),
                                  Variable(zeros(shape=(h3_layer,)), name="F_3/B")],

                          'OUTPUT': [Variable(random.uniform(minval=-math.sqrt(6/(h3_layer + n_actions)), maxval=math.sqrt(6/(h3_layer + n_actions)), shape=(h3_layer, n_actions)), name="OUTPUT/W"),
                                     Variable(zeros(shape=(n_actions, )), name="OUTPUT/B")]}

        self.__trainable = self.__get_weights()

        with open("Config/Model.json") as config:
            self.__model_base = model_from_json(config.read())

    @property
    def weights(self):
        return self.__weights

    @property
    def trainable(self):
        return self.__trainable

    @property
    def optimizer(self):
        return self.__optimizer

    def fullyConnected(self, data):
        with name_scope("FC_1") as scope:
            H_01_INPUT = math.add(matmul(data, self.weights['F_1'][0]), self.weights['F_1'][1])
            H_01_OUTPUT = nn.relu(H_01_INPUT)

        with name_scope("FC_2") as scope:
            H_02_INPUT = math.add(matmul(H_01_OUTPUT, self.weights['F_2'][0]), self.weights['F_2'][1])
            H_02_OUTPUT = nn.relu(H_02_INPUT)

        with name_scope("FC_3") as scope:
            H_03_INPUT = math.add(matmul(H_02_OUTPUT, self.weights['F_3'][0]), self.weights['F_3'][1])
            H_03_OUTPUT = nn.relu(H_03_INPUT)

        with name_scope("OUTPUT") as scope:
            OUTPUT = math.add(matmul(H_03_OUTPUT, self.weights['OUTPUT'][0]), self.weights['OUTPUT'][1])

        return OUTPUT, [H_01_OUTPUT, H_02_OUTPUT, H_03_OUTPUT, OUTPUT]

    @staticmethod
    def loss(y_true, y_pred):
        with name_scope("LOSS") as scope:
            MSE = losses.mean_squared_error(y_true, y_pred)
            LOSS = math.reduce_mean(MSE)

        return LOSS

    @function(input_signature=[TensorSpec(shape=None, dtype=float32), TensorSpec(shape=None, dtype=float32)])
    def training(self, x_value, y_value):
        with GradientTape() as tape:
            tape.watch(self.trainable)

            y_pred = self.fullyConnected(x_value)[0]
            f_loss = self.loss(y_value, y_pred)

        grads = tape.gradient(f_loss, self.trainable)
        self.optimizer.apply_gradients(zip(grads, self.trainable))

    def train(self, x_value, y_value):
        self.training(constant(x_value), constant(y_value))

    @function(input_signature=[TensorSpec(shape=None, dtype=float32)])
    def fastPredict(self, x_value):
        return self.fullyConnected(x_value)

    def predict(self, x_value, learning=False):
        output = self.fastPredict(constant(x_value))

        if not learning:
            self.__activation = output[1]

        return output[0].numpy()

    def __get_weights(self):
        weights = []

        for key in self.__weights.keys():
            weights.extend(self.__weights[key])

        return weights

    def getActivation(self):

        if len(self.__activation) > 0:
            aux = []

            for i, k in enumerate(self.__weights.keys()):
                if k == "F_1":
                    aux.append(self.__activation[i][0].numpy())
                else:
                    aux.append([self.__activation[i][0].numpy(), math.multiply(transpose(self.__activation[i-1]), self.weights[k][0]).numpy()])
        else:
            aux = [[], [], [], []]

        return aux

    def load_model(self, file):
        self.__model_base = load_model(file)

        for i, k in zip(range(len(self.__model_base.layers)), self.__weights.keys()):
            w, b = self.__model_base.layers[i].get_weights()

            if k == "OUTPUT":
                self.__weights[k] = [Variable(w, name="OUTPUT/W"), Variable(b, name=f"OUTPUT/B")]
            else:
                self.__weights[k] = [Variable(w, name=f"F_{i}/W"), Variable(b, name=f"F_{i}/B")]

        self.__trainable = self.__get_weights()

    def save(self, file):
        for i, k in zip(range(len(self.__model_base.layers)), self.__weights.keys()):
            self.__model_base.layers[i].set_weights([self.__weights[k][0].numpy(), self.__weights[k][1].numpy()])

        self.__model_base.save(file)
