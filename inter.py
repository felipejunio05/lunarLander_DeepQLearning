import gym

from tkinter import *
from gym.envs.classic_control import rendering
from numpy import array


# from PIL import ImageTk, Image


class Node(Canvas):
    def __init__(self, master, **kw):
        Canvas.__init__(self, master, **kw)

        self.configure(borderwidth=0, highlightthickness=0)
        self.place(x=0, y=0)

        self.img_model = PhotoImage(file="Images/model.png")
        self.img_activation = PhotoImage(file="Images/n_activate.png")
        self.create_image(0, 0, image=self.img_model, anchor="nw")

        self.connections = self.drawConnections()

        self.H_01_R = [[188, 32], [188, 97], [188, 164], [188, 232], [189, 295], [190, 366]]
        self.H_02_R = [[291, 32], [288, 100], [290, 163], [289, 232], [291, 297], [291, 367]]
        self.OUT_R = [[378, 100], [381, 162], [380, 232], [382, 297]]

        self.H_01_O = [None, None, None, None, None, None]
        self.H_02_O = [None, None, None, None, None, None]
        self.OUT_O = [None, None, None, None]

    def activate(self, lh1=[], lh2=[], lout=[]):

        for i in range(len(lh1)):
            if lh1[i] > 0.56:
                if self.H_01_O[i] is None:
                    self.H_01_O[i] = self.create_image(*self.H_01_R[i], image=self.img_activation)
            else:
                if self.H_01_O[i] is not None:
                    self.delete(self.H_01_O[i])
                    self.H_01_O[i] = None

        for i in range(len(lh2[0])):
            if lh2[0][i] > 0.56:
                if self.H_02_O[i] is None:
                    self.H_02_O[i] = self.create_image(*self.H_02_R[i], image=self.img_activation)
            else:
                if self.H_02_O[i] is not None:
                    self.delete(self.H_02_O[i])
                    self.H_02_O[i] = None

        for i in range(len(lh2[1][0])):
            for j in range(len(lh2[1][1])):
                if lh2[1][i][j] > 0:
                    self.itemconfig(self.connections[0][i][j], fill="red")
                else:
                    self.itemconfig(self.connections[0][i][j], fill="gray")

        for i in range(len(lout[0])):
            if lout[0][i] > 0.56:
                if self.OUT_O[i] is None:
                    self.OUT_O[i] = self.create_image(*self.OUT_R[i], image=self.img_activation)
            else:
                if self.OUT_O[i] is not None:
                    self.delete(self.OUT_O[i])
                    self.OUT_O[i] = None

        for i in range(len(lout[1][0])):
            for j in range(len(lout[1][1])):
                if lout[1][i][j] > 0:
                    self.itemconfig(self.connections[1][i][j], fill="red")
                else:
                    self.itemconfig(self.connections[1][i][j], fill="gray")

    def drawConnections(self):
        conections_0 = []

        for i in range(len(self.H_01_R)):
            conections_0 = []
            for j in range(len(self.H_02_R)):
                conections_0[i].append([self.create_line(self.H_01_R[i][0] + 20, self.H_01_R[i][1],
                                                         self.H_02_R[j][0] - 20, self.H_02_R[j][1], fill="gray")])

        conections_1 = []

        for i in range(len(self.H_02_R)):
            conections_1 = []
            for j in range(len(self.OUT_R)):
                conections_1[i].append(self.create_line(self.H_02_R[i][0] + 20, self.H_02_R[i][1], self.OUT_R[j][0] - 20, self.OUT_R[j][1], fill="gray"))

        return conections_0, conections_1

root = Tk()
root.geometry("1070x400")

frame01 = Frame(root, bg="black", width=600, height=400)
frame01.place(relx=0, rely=0)

frame02 = Frame(root, bg="black", width=470, height=400)
frame02.place(x=600, y=0)

canvas = Node(frame02, width=470, height=399)
canvas.drawConections()

root.mainloop()


def disableViewGym():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def gymRender():
    imgGym = env.render(mode="rgb_array")

    img = Image.fromarray(imgGym)
    imgtk = ImageTk.PhotoImage(image=img)

    screen.imgtk = imgtk
    screen.configure(image=imgtk, borderwidth=0, highlightthickness=0)

    root.after(1, gymRender)


# screen = Label(frame01)
# screen.place(x=0, y=0)


# canvas = Canvas(frame02, width=471, height=400)
# self.configure(borderwidth=0, highlightthickness=0)
#
#
# canvas.place(x=0, y=0)
#
# img_model = PhotoImage(file="Images/model.png")
# canvas.create_image(0, 0, image=img_model, anchor="nw")
#
# activation = PhotoImage(file="Images/n_activate.png")
# canvas.create_image(0, 0)


# canvas.activate([1, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 0], [0, 1, 1, 0])
env = gym.make("LunarLander-v2")
#
# disableViewGym()
# # gymRender()
