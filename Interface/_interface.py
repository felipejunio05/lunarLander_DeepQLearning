from tkinter import Tk
from tkinter import Label
from tkinter import Frame
from tkinter import Canvas
from tkinter import TclError

from PIL import Image
from PIL import ImageTk

from ._draw import Draw

__all__ = ["App"]


class App(Tk):
    def __init__(self):
        super().__init__()

        w = 1200
        h = 920

        w_s = self.winfo_screenwidth()
        h_s = self.winfo_screenheight()

        x, y = ((h_s // 2) - (h // 2) - 20), ((w_s // 2) - (w // 2) - 30)
        self.geometry(f"{w}x{h}+{x}+{y}")

        self.title("Lunar Lander - Deep Q-Learning")

        self.__first = Frame(self, bg="white", width=600, height=400)
        self.__first.place(relx=0, rely=0)

        self.__second = Frame(self, width=600, height=920)
        self.__second.place(x=600, y=0)

        self.__third = Frame(self, bg="white", width=600, height=520)
        self.__third .place(x=0, y=400)

        self.frame_gym = Label(self.__first)
        self.frame_gym.place(x=0, y=0)

        self.__num_episode = Canvas(self.__first, width=100, height=32, bg="black",  borderwidth=0, highlightthickness=0)
        self.__num_episode.place(x=0, y=0)
        self.__num_episode_id = self.__num_episode.create_text(0, 0, font=("Arial", 32), text="01", fill="green", anchor="nw")

        self.__frame_model = Draw(self.__second, width=600, height=920)

        self.__frame_graph = Label(self.__third, width=600, height=520, bg="white")
        self.__frame_graph.place(x=0, y=0)

        self.__pid_render = [None, None]

        self.__agentActivation = [[], [], [], []]
        self.__gymRender = None

        self.withdraw()

    @property
    def agentActivation(self):
        return self.__agentActivation

    @agentActivation.setter
    def agentActivation(self, value):
        self.__agentActivation = value

    @property
    def gymRender(self):
        return self.__gymRender

    @gymRender.setter
    def gymRender(self, value):
        self.__gymRender = value

    def episode(self, string):
        self.__num_episode.delete(self.__num_episode_id)
        self.__num_episode_id = self.__num_episode.create_text(0, 0, font=("Arial", 32), text=string, fill="green", anchor="nw")

    def close(self):
        self.destroy()
        self.after_cancel(self.__pid_render[0])
        self.after_cancel(self.__pid_render[1])

    def frameUpdater1(self):
        try:
            if self.__gymRender is not None:
                imgar = Image.fromarray(self.__gymRender)
                imgtk = ImageTk.PhotoImage(image=imgar)

                self.frame_gym.imgtk = imgtk
                self.frame_gym.configure(image=imgtk, borderwidth=0, highlightthickness=0)
        except TclError:
            pass

        self.__pid_render[0] = self.after(1, func=self.frameUpdater1)

    def frameUpdater2(self):
        try:
            if len(self.__agentActivation[0]) > 0:
                self.__frame_model.activate(*self.__agentActivation)
        except TclError:
            pass

        self.__pid_render[1] = self.after(50, func=self.frameUpdater2)

    def frameUpdater3(self, img):
        imgar = Image.fromarray(img).resize(size=(600, 520), resample=Image.ADAPTIVE)
        imgtk = ImageTk.PhotoImage(image=imgar)

        self.__frame_graph.imgtk = imgtk
        self.__frame_graph.configure(image=imgtk, borderwidth=0, highlightthickness=0)

    def jobs(self):
        self.after(1, self.frameUpdater1())
        self.after(50, self.frameUpdater2())
