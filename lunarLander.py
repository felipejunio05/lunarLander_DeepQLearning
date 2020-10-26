import gym

from DQN import Agent
from Graph import Graph
from Interface import App

from threading import Thread
from utils import disableViewGym

if __name__ == "__main__":
    disableViewGym()
    environment = gym.make("LunarLander-v2")

    lr = 0.001
    EPSILON = 1.0
    EPISODES = 700
    GAMMA = 0.99
    MEM_SIZE = 1000000
    BATCH_SIZE = 64

    INPUT_DIMS = environment.observation_space.shape[0]
    ACTIONS_SPACE = environment.action_space.n

    SAVE_MODEL_DIR = "Model/dqn_model.h5"
    SAVE_GRAPH_DIR = "Results/agent.png"

    app = App()
    graph = Graph(SAVE_GRAPH_DIR)
    agent = Agent(lr, GAMMA, EPSILON, ACTIONS_SPACE, INPUT_DIMS, BATCH_SIZE, mem_size=MEM_SIZE, saveIn=SAVE_MODEL_DIR)

    th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph), daemon=True)
    th_train.start()

    app.jobs()
    app.mainloop()
