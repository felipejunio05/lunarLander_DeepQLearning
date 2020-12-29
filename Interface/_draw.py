from numpy import absolute as npy_absolute
from tkinter import Canvas
from tkinter import PhotoImage


__all__ = ["Draw"]


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
        self.connections = [[], [], []]
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
                self.activationVertex(i, 0, lh2[1], True)

            elif (lh1[i] > 0) and (lh1[i] < 0.56):
                self.drawActivation(*self.nodes[0][i], 1)
                self.activationVertex(i, 0, lh2[1], True)

            else:
                self.drawActivation(*self.nodes[0][i], 2)
                self.activationVertex(i, 0, lh2[1], False)

            if lh2[0][i] >= 0.56:
                self.drawActivation(*self.nodes[1][i], 0)
                self.activationVertex(i, 1, lh3[1], True)

            elif (lh2[0][i] > 0) and (lh2[0][i] < 0.56):
                self.drawActivation(*self.nodes[1][i], 1)
                self.activationVertex(i, 1, lh3[1], True)

            else:
                self.drawActivation(*self.nodes[1][i], 2)
                self.activationVertex(i, 1, lh3[1], False)

            if lh3[0][i] >= 0.56:
                self.drawActivation(*self.nodes[2][i], 0)
                self.activationVertex(i, 2, lout[1], True, True)

                LAYER_3 = True

            elif (lh3[0][i] > 0) and (lh3[0][i] < 0.56):
                self.drawActivation(*self.nodes[2][i], 1)

            else:
                self.drawActivation(*self.nodes[2][i], 2)
                self.activationVertex(i, 2, lout[1], False)

        if LAYER_3:
            IDX_OUTPUT = lout[0].argmax()

            if self.__auxOutput is not None:
                self.drawActivation(*self.nodes[3][self.__auxOutput], 2)

            if lout[0][IDX_OUTPUT] >= 0.56:
                self.drawActivation(*self.nodes[3][IDX_OUTPUT], 0)

            else:
                self.drawActivation(*self.nodes[3][IDX_OUTPUT], 1)

            self.__auxOutput = IDX_OUTPUT

    def drawConnections(self):
        for i in range(len(self.nodes) - 1):
            for j in range(len(self.nodes[i])):
                self.connections[i].append([])

                for k in range(len(self.nodes[i + 1])):
                    if i != 2:
                        self.connections[i][j].append(
                            self.create_line(self.nodes[i][j][0] + 7, self.nodes[i][j][1], self.nodes[i + 1][k][0] - 5,
                                             self.nodes[i + 1][k][1], fill="gray"))
                    else:
                        self.connections[i][j].append(
                            self.create_line(self.nodes[i][j][0] + 7, self.nodes[i][j][1], self.nodes[i + 1][k][0] - 5,
                                             self.nodes[i + 1][k][1], fill="gray"))

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
            if self.itemcget(self.activation[0][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[0][(x, y)], state="hidden")

            if self.itemcget(self.activation[1][(x, y)], "state") == "hidden":
                self.itemconfig(self.activation[1][(x, y)], state="normal")

        elif options == 1:
            if self.itemcget(self.activation[1][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[1][(x, y)], state="hidden")

            if self.itemcget(self.activation[0][(x, y)], "state") == "hidden":
                self.itemconfig(self.activation[0][(x, y)], state="normal")

        elif options == 2:
            if self.itemcget(self.activation[0][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[0][(x, y)], state="hidden")

            if self.itemcget(self.activation[1][(x, y)], "state") == "normal":
                self.itemconfig(self.activation[1][(x, y)], state="hidden")

    def activationVertex(self, neuron, layer, values, enable=True, output=False):
        if enable:
            for j in range(values.shape[1]):
                if not output:
                    if values[neuron][j] > 0:
                        if self.itemcget(self.connections[layer][neuron][j], "fill") == "gray":
                            self.itemconfig(self.connections[layer][neuron][j], fill="red")
                else:
                    if npy_absolute(values[neuron][j]) > 0.56:
                        if self.itemcget(self.connections[layer][neuron][j], "fill") == "gray":
                            self.itemconfig(self.connections[layer][neuron][j], fill="red")
        else:
            for j in range(values.shape[1]):
                if self.itemcget(self.connections[layer][neuron][j], "fill") == "red":
                    self.itemconfig(self.connections[layer][neuron][j], fill="gray")
