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
    Caso contrário basta acessar o arquivo lunarLander.py e alteralos conforme as suas necessidades, veja os exemplos abaixo:
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
th_train = Thread(target=agent.run, args=(EPISODES, environment, graph=graph), daemon=True)
```

<p>
   Sem geração de gráfico:
</p>

```python
# com gráfico
th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph), daemon=True)

# sem gráfico
th_train = Thread(target=agent.run, args=(EPISODES, environment, app), daemon=True)
```

<p>
   Sem gráfico e interface:
</p>

```python
# com gráfico e interface
th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph), daemon=True)

# sem gráfico e interface
th_train = Thread(target=agent.run, args=(EPISODES, environment), daemon=True)
```

<p>
   Carregando um modelo salvo:
</p>

```python
LOAD_MODEL = "Model/dqn_model.h5"

th_train = Thread(target=agent.run, args=(EPISODES, environment, app, graph, LOAD_MODEL), daemon=True)
```

<h2>DEMO</h2>

<strong>Link:</strong> https://youtu.be/v00rulenDrY 
