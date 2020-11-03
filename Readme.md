<h1>LUNAR LANDER - DEEP Q-LEARNING</h1>

Implementação do OpenAI Gym Lunar Lander com Deep Q-Learning, o objetivo do agente é pousar o modulo lunar na região delimitada pelas bandeiras.

<p align="center"><img src="Model/model.gif"></p>

<h2>HIPERPARÂMETROS</h2>

<h3>Agente</h3>

<ul>
    <li>EPSILON -> 1.0</li>
    <li>EPSILON_D -> 0.01</li>
    <li>EPSILON_M -> 0.001</li>
    <li>EPISÓDIOS -> 700</li>
    <li>GAMMA -> 0.99</li>
    <li>TX APRENDIZAGEM -> 0.001</li>
    <li>TAMANHO DA MEMÓRIA -> 1000000</li>
    <li>TAMANHO DO LOTE -> 64</li>
</ul>

<h3>Rede Neural</h3>
<ul>
    <li> 1 -> Camada de entrada com 8 neuronios </li>
    <li> 3 -> Camadas ocultas com 32 neuronios com bias 0. f: relu </li>
    <li> 1 -> Camada de saidas com 4 neuronios com bias 0.  f: linear </li>
</ul>

<h2>DESEMPENHO/TREINAMENTO</h2>
<p align="center"><img src="Results/agent.png"></p>

<h2>PRÉ-REQUISITOS</h2>
<ul>
    <li>Python -> 3.8</li>
    <li>Pacotes -> requirements.txt</li>
</ul>
 
<h2>COMO EXECUTAR</h2>

<p>
    Para executar com os Hiperparâmetros listados acima, basta ir no terminal, no diretorio do script e executar comando:
</p>

```shell
python lunarLander.py
```

<p>
    Caso contrário, basta acessar o arquivo lunarLander.py e alteralos conforme as suas necessidades, veja os exemplos abaixo:
</p>

<p>
    Alteração dos Hiperparâmetros:
</p>


```python
lr = 'change here' # Taxa de Aprendizagem
EPSILON = 'change here'
EPSILON_MIN = 'change here'
EPSILON_DECAY = 'change here'
EPISODES = 'change here'
GAMMA = 'change here'
MEM_SIZE = 'change here'
BATCH_SIZE = 'change here'
```

<p>
    Alteração do caminho onde o gráfico e o modelo serão salvos:
<p>

```python
SAVE_MODEL_DIR = 'change here'
SAVE_GRAPH_DIR = 'change here'
```

<p>
    Executando o treinamento sem interface:
</p>

```python
# com interface
th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph), daemon=True)

# sem interface
#app = App()

agent.run(EPISODES, environment, graph=graph)

#th_train.start()

#app.jobs()
#app.mainloop()
```

<p>
   Sem geração de gráfico:
</p>

```python
# com gráfico
th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph), daemon=True)

# sem gráfico

#SAVE_GRAPH_DIR = "Results/agent.png"
#graph = Graph(SAVE_GRAPH_DIR)

th_train = Thread(target=agent.run, args=(EPISODES, environment, app), daemon=True)
```

<p>
   Sem gráfico e interface:
</p>

```python
# com gráfico e interface
th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph), daemon=True)

# sem gráfico e interface

#SAVE_GRAPH_DIR = "Results/agent.png"

#app = App()
#graph = Graph(SAVE_GRAPH_DIR)

agent.run(EPISODES, environment)

#th_train.start()

#app.jobs()
#app.mainloop()
```

<p>
   Carregando um modelo salvo:
</p>

```python
LOAD_MODEL = "Model/dqn_model.h5"

th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph, LOAD_MODEL), daemon=True)
```

<p>
   Também é possivel customizar o treinamento, já que os métodos utilizados no 'agent.run()' estão públicos: 
</p>

```python
import gym
from DQN import Agent


lr = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.001
EPISODES = 700
GAMMA = 0.99
MEM_SIZE = 1000000
BATCH_SIZE = 64

env = gym.make("LunarLander-v2")

INPUT_DIMS = env.observation_space.shape[0]
ACTIONS_SPACE = env.action_space.n

SAVE_MODEL_DIR = "Model/dqn_model.h5"

agent = Agent(lr, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, ACTIONS_SPACE, INPUT_DIMS, BATCH_SIZE, MEM_SIZE, SAVE_MODEL_DIR)

for episode in range(EPISODES):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, info = env.step(action)

        agent.store_transition(state, action, reward, new_state, done)
        agent.learn()

        state = new_state
```
<h2>DEMO</h2>

<strong>Link:</strong> https://youtu.be/v00rulenDrY 
