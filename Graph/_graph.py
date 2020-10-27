from numpy import array as npy_array
from numpy import arange as npy_arange
from numpy import frombuffer as npy_frombuffer

import matplotlib.pyplot as plt
from matplotlib import use as plot_backend

plot_backend("Agg")

__all__ = ["Graph"]


class Graph:
    def __init__(self, saveIn="graph.png"):
        self.__figure = plt.figure(figsize=(8, 8))

        self.__ax_1 = self.__figure.add_subplot(label="1")
        self.__ax_2 = self.__figure.add_subplot(label="2", frame_on=False)

        self.__ax_1.set_title("Desempenho do Agente")
        self.__ax_1.set_xlabel("Episódios", color="k")
        self.__ax_1.set_ylabel("Epsilon", color="k")
        self.__ax_1.tick_params(axis="x", colors="k")
        self.__ax_1.tick_params(axis="y", colors="k")

        self.__ax_2.axes.get_xaxis().set_visible(False)
        self.__ax_2.yaxis.tick_right()

        self.__ax_2.set_ylabel("Pontuação", color="k")
        self.__ax_2.yaxis.set_label_position("right")
        self.__ax_2.tick_params(axis="y", colors="k")

        self.__saveIn = saveIn

    @property
    def saveIn(self):
        return self.__saveIn

    def plot(self, x, y1, y2, start=False):
        if not start:
            x = npy_arange(x + 1)

            self.__ax_1.plot(x, npy_array(y1), color="c")
            self.__ax_2.scatter(x, npy_array(y2), color="r")

            img = self.__convertFigToArray()

        else:
            img = self.__convertFigToArray()

        return img

    def __convertFigToArray(self):
        self.__figure.canvas.draw()

        img = npy_frombuffer(self.__figure.canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(self.__figure.canvas.get_width_height()[::-1] + (3,))

        return img

    def save(self):
        self.__figure.set_figwidth(10)
        self.__figure.set_figheight(8)

        self.__figure.savefig(self.__saveIn)
