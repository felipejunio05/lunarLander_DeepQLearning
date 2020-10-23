def train():
    global graphic
    global gym_render
    global agent_activation

    disableViewGym()
    env = gym.make("LunarLander-v2")

    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n, mem_size=1000000, batch_size=128, fname="Model/dqn_model_01.h5")

    if os.path.exists("Model/dqn_model.h5"):
        agent.load_model()

    scores = []
    eps_history = []

    for i in range(500):
        state = env.reset()

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
            avg_score = np.mean(scores[-100:])
            eps_history.append(agent.epsilon)

            state = new_state
            print('episode: %i' % i, 'scores %.2f' % score, 'average_score %.2f' % avg_score,
                  'epsilon %2f' % agent.epsilon)

    agent.save_model()
    app.after_cancel(pid_train)
    app.destroy()


def disableViewGym():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


def frameUpdater():
    global pid_render
    global graphic

    if gym_render is not None:
        imgar = Image.fromarray(gym_render)
        imgtk = ImageTk.PhotoImage(image=imgar)

        app.frame_gym.imgtk = imgtk
        app.frame_gym.configure(image=imgtk, borderwidth=0, highlightthickness=0)

    if len(agent_activation[0]) > 0:
        app.frame_model.activate(*agent_activation)

    pid_render = app.after(1, func=frameUpdater)


if __name__ == "__main__":
    import os
    import numpy as np
    import gym
    import threading

    from DQN import Agent
    from Interface import App
    from threading import Thread
    from PIL import Image, ImageTk

    agent_activation = [[], [], []]

    graphic = [None, True]
    gym_render = None
    pid_render = None

    app = App()

    th_train = Thread(target=train, args=(), daemon=True)
    th_train.start()

    frameUpdater()
    app.mainloop()
