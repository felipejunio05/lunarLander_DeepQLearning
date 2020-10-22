from tkinter import Tk
from tkinter import Label
from tkinter import Frame
from tkinter import Canvas
from tkinter import PhotoImage

from Interface import H1_NEURON_SCREEN_POS
from Interface import H2_NEURON_SCREEN_POS
from Interface import OT_NEURON_SCREEN_POS

__all__ = ["App"]


class Node(Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)

        self.configure(borderwidth=0, highlightthickness=0)
        self.place(x=0, y=0)

        self.img_activation_01 = PhotoImage(file="Images/activation_01.png")
        self.img_activation_02 = PhotoImage(file="Images/activation_02.png")
        self.img_activation_03 = PhotoImage(file="Images/activation_03.png")

        self.img_model = PhotoImage(file="Images/model.png")
        self.create_image(0, 0, image=self.img_model, anchor="nw")

        self.connections = self.drawConnections()

        self.H_01_O = [None, None, None, None, None, None]
        self.H_02_O = [None, None, None, None, None, None]
        self.OUT_O = [None, None, None, None]

    def activate(self, lh1=[], lh2=[], lout=[]):
        for i in range(lh1.shape[0]):
            if lh1[i] >= 0.56:
                if self.H_01_O[i] is None:
                    self.H_01_O[i] = self.drawActivation(*H1_NEURON_SCREEN_POS[i], False)
                    self.activationVertex(i, 0, lh2[1])
                else:
                    self.delete(self.H_01_O[i])
                    self.activationVertex(i, 0, lh2[1], False)

                    self.H_01_O[i] = self.drawActivation(*H1_NEURON_SCREEN_POS[i], False)
                    self.activationVertex(i, 0, lh2[1])

            elif (lh1[i] > 0) and (lh1[i] < 0.56):
                if self.H_01_O[i] is None:
                    self.H_01_O[i] = self.drawActivation(*H1_NEURON_SCREEN_POS[i], True)
                    self.activationVertex(i, 0, lh2[1])
                else:
                    self.delete(self.H_01_O[i])
                    self.activationVertex(i, 0, lh2[1], False)

                    self.H_01_O[i] = self.drawActivation(*H1_NEURON_SCREEN_POS[i], True)
                    self.activationVertex(i, 0, lh2[1])
            else:
                if self.H_01_O[i] is not None:
                    self.delete(self.H_01_O[i])
                    self.activationVertex(i, 0, lh2[1], False)

                    self.H_01_O[i] = None

        H2_ACTIVATE = False

        for i in range(lh2[0].shape[0]):
            if lh2[0][i] >= 0.56:
                H2_ACTIVATE = True

                if self.H_02_O[i] is None:
                    self.H_02_O[i] = self.drawActivation(*H2_NEURON_SCREEN_POS[i], False)
                    self.activationVertex(i, 1, lout[1])

                else:
                    self.delete(self.H_02_O[i])
                    self.activationVertex(i, 1, lout[1], False)

                    self.H_02_O[i] = self.drawActivation(*H2_NEURON_SCREEN_POS[i], False)
                    self.activationVertex(i, 1, lout[1])

            elif (lh2[0][i] > 0) and (lh2[0][i]) < 0.56:
                H2_ACTIVATE = True

                if self.H_02_O[i] is None:
                    self.H_02_O[i] = self.drawActivation(*H2_NEURON_SCREEN_POS[i], True)
                    self.activationVertex(i, 1, lout[1])

                else:
                    self.delete(self.H_02_O[i])
                    self.activationVertex(i, 1, lout[1], False)

                    self.H_02_O[i] = self.drawActivation(*H2_NEURON_SCREEN_POS[i], True)
                    self.activationVertex(i, 1, lout[1])

            else:
                if self.H_02_O[i] is not None:
                    self.delete(self.H_02_O[i])
                    self.activationVertex(i, 1, lout[1], False)

                    self.H_02_O[i] = None

        idx_output = lout[0].argmax()

        for i in range(len(lout[0])):
            if self.OUT_O[i] is not None:
                self.activationVertex(i, 1, lout[1], False)
                self.delete(self.OUT_O[i])

                self.OUT_O[i] = None

        if H2_ACTIVATE:
            self.OUT_O[idx_output] = self.drawActivation(*OT_NEURON_SCREEN_POS[idx_output], False)

    def drawConnections(self):
        conections_0 = []

        for i in range(len(H1_NEURON_SCREEN_POS)):
            conections_0.append([])

            for j in range(len(H2_NEURON_SCREEN_POS)):
                conections_0[i].append([self.create_line(H1_NEURON_SCREEN_POS[i][0] + 20, H1_NEURON_SCREEN_POS[i][1],
                                                         H2_NEURON_SCREEN_POS[j][0] - 20, H2_NEURON_SCREEN_POS[j][1], fill="gray")])

        conections_1 = []

        for i in range(len(H2_NEURON_SCREEN_POS)):
            conections_1.append([])

            for j in range(len(OT_NEURON_SCREEN_POS)):
                conections_1[i].append(self.create_line(H2_NEURON_SCREEN_POS[i][0] + 20, H2_NEURON_SCREEN_POS[i][1],
                                                        OT_NEURON_SCREEN_POS[j][0] - 20, OT_NEURON_SCREEN_POS[j][1], fill="gray"))

        return conections_0, conections_1

    def drawActivation(self, x, y, partial):
        if not partial:
            objId = self.create_image(x, y, image=self.img_activation_01)
            self.delete(objId)

            objId = self.create_image(x, y, image=self.img_activation_02)
            self.delete(objId)

            objId = self.create_image(x, y, image=self.img_activation_03)
        else:
            objId = self.create_image(x, y, image=self.img_activation_01)

        return objId

    def activationVertex(self, neuron, layer, values, enable=True):

        if values != []:
            if enable:
                for j in range(values.shape[1]):
                    if values[neuron][j] > 0:
                        self.itemconfig(self.connections[layer][neuron][j], fill="red")
            else:
                for j in range(values.shape[1]):
                    self.itemconfig(self.connections[layer][neuron][j], fill="gray")


class App(Tk):

    def __init__(self):
        super().__init__()

        self.geometry("1070x400")
        #self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.title("Lunar Lander - Deep Q-Learning")

        self.primary = Frame(self, width=600, height=400)
        self.primary.place(relx=0, rely=0)

        self.secondary = Frame(self, bg="black", width=470, height=400)
        self.secondary.place(x=600, y=0)

        self.frame_gym = Label(self.primary)
        self.frame_gym.place(x=0, y=0)

        self.frame_model = Node(self.secondary, width=470, height=399)
