import gym

from tkinter import *
from gym.envs.classic_control import rendering
from numpy import array



import numpy as np


root = Tk()
root.geometry("1200x920")

frame01 = Frame(root, bg="white", width=600, height=920)
frame01.place(relx=0, rely=0)

frame02 = Frame(root, bg="black", width=600, height=920)
frame02.place(x=600, y=0)

canvas = Node(frame02, width=600, height=920)

canvas.activate(np.random.uniform(-1, 1, size=(32, )), [np.random.uniform(-1, 1, size=(32, )), np.random.uniform(-1, 1, size=(32, 32))],
                [np.random.uniform(-1, 1, size=(32, )), np.random.uniform(-1, 1, size=(32, 32))], [np.random.uniform(-1, 1, size=(4, )),
                                                                                               np.random.uniform(-1, 1, size=(32, 4))])
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
