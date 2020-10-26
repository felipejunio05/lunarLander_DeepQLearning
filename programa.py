import os
import gym

from DQN import Agent
from Interface import App

from threading import Thread
from tkinter import TclError
from PIL import Image, ImageTk

from numpy import mean as npy_mean
from numpy import array as npy_array
from numpy import arange as npy_arange
from numpy import frombuffer as npy_frombuffer


from matplotlib import use as plot_backend
plot_backend("Agg")

import matplotlib.pyplot as plt


def train():
    global gym_render
    global agent_activation

    disableViewGym()
    env = gym.make("LunarLander-v2")

    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n, mem_size=1000000, batch_size=64, fname="Model/dqn_model.h5")

    if os.path.exists("Model/dqn_model.h5"):
        agent.load_model()

    avg_score = []
    figure = graph()

    frameUpdater3(convertFigToArray(figure))
    app.deiconify()

    for i in range(700):
        state = env.reset()

        scores = []
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            agent_activation = agent.q_eval.getActivation()

            new_state, reward, done, info = env.step(action)
            gym_render = env.render(mode="rgb_array")

            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()

            score += reward
            scores.append(score)

            state = new_state

        app.episode(str(i+1).zfill(2))
        avg_score.append(npy_mean(scores[i]))

        if i > 50:
            plt.plot(npy_arange(i + 1), npy_array(avg_score), "r")
            frameUpdater3(convertFigToArray(figure))

    agent.save_model()
    app.close(*pid_render)


def graph():

    fig = plt.figure(figsize=(6, 5))

    plt.title('Desempenho do Agente')
    plt.xlabel('Episódios')
    plt.ylabel('Pontuação')

    return fig


def convertFigToArray(fig):
    fig.canvas.draw()

    img = npy_frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img


def disableViewGym():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def frameUpdater1():
    global pid_render

    try:
        if gym_render is not None:
            imgar = Image.fromarray(gym_render)
            imgtk = ImageTk.PhotoImage(image=imgar)

            app.frame_gym.imgtk = imgtk
            app.frame_gym.configure(image=imgtk, borderwidth=0, highlightthickness=0)
    except TclError:
        pass

    pid_render[0] = app.after(1, func=frameUpdater1)


def frameUpdater2():
    global pid_render

    try:
        if len(agent_activation[0]) > 0:
            app.frame_model.activate(*agent_activation)
    except TclError:
        pass

    pid_render[1] = app.after(50, func=frameUpdater2)


def frameUpdater3(img):

    imgar = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=imgar)

    app.frame_graph.imgtk = imgtk
    app.frame_graph.configure(image=imgtk, borderwidth=0, highlightthickness=0)


if __name__ == "__main__":
    agent_activation = [[], [], [], []]

    gym_render = None
    pid_render = [None, None]

    app = App()

    th_train = Thread(target=train, args=(), daemon=True)
    th_train.start()

    app.after(1, frameUpdater1)
    app.after(50, frameUpdater2)
    app.mainloop()
