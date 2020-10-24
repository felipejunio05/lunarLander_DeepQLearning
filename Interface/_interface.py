from numpy import absolute
from tkinter import Tk
from tkinter import Label
from tkinter import Frame
from tkinter import Canvas
from tkinter import PhotoImage


__all__ = ["App"]


class Draw(Canvas):
    def __init__(self, master, **kw):
        Canvas.__init__(self, master, **kw)

        self.configure(borderwidth=0, highlightthickness=0)
        self.place(x=0, y=0)

        self.img_disable = PhotoImage(file="Images/disable.png")
        self.img_enable_1 = PhotoImage(file="Images/enable_1.png")
        self.img_enable_2 = PhotoImage(file="Images/enable_2.png")
        self.img_model = PhotoImage(file="Images/background.png")

        self.nodes = []
        self.connections = []
        self.activation = [{}, {}]

        self.__auxOutput = None

        self.create_image(0, 0, image=self.img_model, anchor="nw")
        self.drawNeurons()
        self.drawConnections()

    def activate(self, lh1, lh2, lh3, lout):
        LAYER_3 = False

        for i in range(lh1.shape[0]):
            if lh1[i] >= 0.56:
                self.drawActivation(*self.nodes[0][i], 0)
                self.activationVertex(i, 0, lh2[1])

            elif (lh1[i] > 0) and (lh1[i] < 0.56):
                self.drawActivation(*self.nodes[0][i], 1)
                self.activationVertex(i, 0, lh2[1])

            else:
                self.drawActivation(*self.nodes[0][i], 2)
                self.activationVertex(i, 0, lh2[1], False)

            if lh2[0][i] >= 0.56:
                self.drawActivation(*self.nodes[1][i], 0)
                self.activationVertex(i, 1, lh3[1])

            elif (lh2[0][i] > 0) and (lh2[0][i] < 0.56):
                self.drawActivation(*self.nodes[1][i], 1)
                self.activationVertex(i, 1, lh3[1])

            else:
                self.drawActivation(*self.nodes[1][i], 2)
                self.activationVertex(i, 1, lh3[1], False)

            if lh3[0][i] >= 0.56:
                self.drawActivation(*self.nodes[2][i], 0)
                self.activationVertex(i, 2, lout[1])

                LAYER_3 = True

            elif (lh3[0][i] > 0) and (lh3[0][i] < 0.56):
                self.drawActivation(*self.nodes[2][i], 1)
                self.activationVertex(i, 2, lout[1])

                LAYER_3 = True
            else:
                self.drawActivation(*self.nodes[2][i], 2)
                self.activationVertex(i, 2, lout[1], False)

        if LAYER_3:
            IDX_OUTPUT = lout[0].argmax()

            if self.__auxOutput is not None:
                self.drawActivation(*self.nodes[3][self.__auxOutput], 2)

            if lout[0][IDX_OUTPUT] >= 0.56:
                self.drawActivation(*self.nodes[3][IDX_OUTPUT], 0)

            elif (lout[0][IDX_OUTPUT] > 0) and (lout[0][IDX_OUTPUT] < 0.56):
                self.drawActivation(*self.nodes[3][IDX_OUTPUT], 1)

            else:
                self.drawActivation(*self.nodes[3][IDX_OUTPUT], 2)

            self.__auxOutput = IDX_OUTPUT

    def drawConnections(self):
        for i in range(len(self.nodes) - 1):
            self.connections.append([])

            for j in range(len(self.nodes[i])):
                self.connections[i].append([])

                for k in range(len(self.nodes[i + 1])):
                    if i + 1 != 3:
                        self.connections[i][j].append(self.create_line(self.nodes[i][j][0] + 175, self.nodes[i][j][1], self.nodes[i][k][0]+7, self.nodes[i][k][1], fill="gray"))
                    else:
                        self.connections[i][j].append(self.create_line(self.nodes[i][j][0]+6, self.nodes[i][j][1], self.nodes[i + 1][k][0]-7,  self.nodes[i + 1][k][1], fill="gray"))

    def drawNeurons(self):
        x = 30

        for i in range(3):
            self.nodes.append([])

            y = 28

            for j in range(32):
                self.create_image(x, y, image=self.img_disable)

                id_1 = self.create_image(x, y, image=self.img_enable_1)
                id_2 = self.create_image(x, y, image=self.img_enable_2)

                self.itemconfig(id_1, state="hidden")
                self.itemconfig(id_2, state="hidden")

                self.activation[0][(x, y)] = id_1
                self.activation[1][(x, y)] = id_2

                self.nodes[i].append([x, y])
                y += 28

            x += 180

        y = 330

        self.nodes.append([])

        for i in range(4, 8):
            self.create_image(x, y, image=self.img_disable)

            id_1 = self.create_image(x, y, image=self.img_enable_1)
            id_2 = self.create_image(x, y, image=self.img_enable_2)

            self.itemconfig(id_1, state="hidden")
            self.itemconfig(id_2, state="hidden")

            self.activation[0][(x, y)] = id_1
            self.activation[1][(x, y)] = id_2

            self.nodes[-1].append([x, y])

            y += 100

    def drawActivation(self, x, y, options):
        if options == 0:
            if self.itemcget(self.activation[1][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[1][(x, y)], state="hidden")

            if self.itemcget(self.activation[0][(x, y)], "state") == "hidden":
                self.itemconfig(self.activation[0][(x, y)], state="normal")

        elif options == 1:
            if self.itemcget(self.activation[0][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[0][(x, y)], state="hidden")

            if self.itemcget(self.activation[1][(x, y)], "state") == "hidden":
                self.itemconfig(self.activation[1][(x, y)], state="normal")

        elif options == 2:
            if self.itemcget(self.activation[0][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[0][(x, y)], state="hidden")

            if self.itemcget(self.activation[1][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[1][(x, y)], state="hidden")

    def activationVertex(self, neuron, layer, values, enable=True):
        if enable:
            for j in range(values.shape[1]):
                if absolute(values[neuron][j]) > 0:
                    if self.itemcget(self.connections[layer][neuron][j], "fill") == "gray":
                        self.itemconfig(self.connections[layer][neuron][j], fill="red")
                else:
                    if self.itemcget(self.connections[layer][neuron][j], "fill") == "red":
                        self.itemconfig(self.connections[layer][neuron][j], fill="gray")
        else:
            for j in range(values.shape[1]):
                if self.itemcget(self.connections[layer][neuron][j], "fill") == "red":
                    self.itemconfig(self.connections[layer][neuron][j], fill="gray")


class App(Tk):

    def __init__(self):
        super().__init__()

        w = 1200
        h = 920

        w_s = self.winfo_screenwidth()  # width of the screen
        h_s = self.winfo_screenheight()

        x, y = ((h_s//2) - (h//2) - 20), ((w_s//2) - (w//2) - 30)
        self.geometry(f"{w}x{h}+{x}+{y}")

        self.title("Lunar Lander - Deep Q-Learning")

        self.primary = Frame(self, bg="white", width=600, height=920)
        self.primary.place(relx=0, rely=0)

        self.secondary = Frame(self, width=600, height=920)
        self.secondary.place(x=600, y=0)

        self.frame_gym = Label(self.primary)
        self.frame_gym.place(x=0, y=0)

        self.frame_model = Draw(self.secondary, width=600, height=920)
