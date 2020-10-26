from tensorflow import nn
from tensorflow import zeros
from tensorflow import random
from tensorflow import matmul
from tensorflow import losses
from tensorflow import Variable
from tensorflow import transpose
from tensorflow import name_scope
from tensorflow import optimizers
from tensorflow import GradientTape
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

__all__ = ["NeuralNetwork"]


class NeuralNetwork:
    def __init__(self, n_states, h1_layer, h2_layer, h3_layer, n_actions, lr):

        self.__optimizer = optimizers.Adam(lr)
        self.__activation = [[], [], [], []]

        self.__weights = {'F_1': [Variable(random.uniform(minval=-1, maxval=1, shape=(n_states, h1_layer)), name="F_1/W"),
                                Variable(zeros(shape=(h1_layer, )), name="F_1/B")],

                        'F_2': [Variable(random.uniform(minval=-1, maxval=1, shape=(h1_layer, h2_layer)), name="F_2/W"),
                                 Variable(zeros(shape=(h2_layer, )), name="F_2/B")],

                        'F_3': [Variable(random.uniform(minval=-1, maxval=1, shape=(h2_layer, h3_layer)), name="F_3/W"),
                                Variable(zeros(shape=(h3_layer,)), name="F_3/B")],

                        'F_4': [Variable(random.uniform(minval=-1, maxval=1, shape=(h3_layer, n_actions)), name="F_4/W"),
                                Variable(zeros(shape=(n_actions, )), name="F_4/B")]}

        with open("Config/Model.json") as config:
            self.__model_base = model_from_json(config.read())

    def __fullyConneted(self, data, train=False):

        with name_scope("FC_1") as scope:
            H_01_INPUT = matmul(data, self.__weights['F_1'][0])
            H_01_OUTPUT = nn.relu(H_01_INPUT + self.__weights['F_1'][1], name="FC_01")

        with name_scope("FC_2") as scope:
            H_02_INPUT = matmul(H_01_OUTPUT, self.__weights['F_2'][0])
            H_02_OUTPUT = nn.relu(H_02_INPUT + self.__weights['F_2'][1], name="FC_02")

        with name_scope("FC_3") as scope:
            H_03_INPUT = matmul(H_02_OUTPUT, self.__weights['F_3'][0])
            H_03_OUTPUT = nn.relu(H_03_INPUT + self.__weights['F_3'][1], name="FC_03")

        with name_scope("FC_4") as scope:
            Y_PRED = matmul(H_03_OUTPUT, self.__weights['F_4'][0]) + self.__weights['F_4'][1]

        if not train:
            self.__activation[0] = H_01_OUTPUT[0].numpy()
            self.__activation[1] = [H_02_OUTPUT[0].numpy(), (transpose(H_01_OUTPUT) * self.__weights['F_2'][0]).numpy()]
            self.__activation[2] = [H_03_OUTPUT[0].numpy(), (transpose(H_02_OUTPUT) * self.__weights['F_3'][0]).numpy()]
            self.__activation[3] = [Y_PRED[0].numpy(), (transpose(H_03_OUTPUT) * self.__weights['F_4'][0]).numpy()]

        return Y_PRED

    @staticmethod
    def __loss(y_true, y_pred):
        return losses.mean_squared_error(y_true, y_pred)

    def train(self, x_value, y_value):
        weights = self.__get_weights()

        with GradientTape() as tape:
            tape.watch(weights)

            y_pred = self.__fullyConneted(x_value, train=True)
            f_loss = self.__loss(y_value, y_pred)

        grads = tape.gradient(f_loss, weights)
        self.__optimizer.apply_gradients(zip(grads, weights))

    def predict(self, x_value, learning=False):
        if learning:
            y_pred = self.__fullyConneted(x_value, train=True).numpy()
        else:
            y_pred = self.__fullyConneted(x_value, train=False).numpy()

        return y_pred

    def __get_weights(self):
        weights = []

        for key in self.__weights.keys():
            weights.extend(self.__weights[key])

        return weights

    def getActivation(self):
        return self.__activation

    def load_model(self, file):
        self.__model_base = load_model(file)

        for i, k in zip(range(len(self.__model_base.layers)), self.__weights.keys()):
            w, b = self.__model_base.layers[i].get_weights()
            self.__weights[k] = [Variable(w, name=f"F_{i}/W"), Variable(b, name=f"F_{i}/B")]

    def save(self, file):
        for i, k in zip(range(len(self.__model_base.layers)), self.__weights.keys()):
            self.__model_base.layers[i].set_weights([self.__weights[k][0].numpy(), self.__weights[k][1].numpy()])

        self.__model_base.save(file)
