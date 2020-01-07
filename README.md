# DRL-Atari

# STATUS : WORK IN PROGRESS

The goal of this repo is to implement RL algorithms and try to benchmarks them in the atari environment made by OpenAI.
For the moment, I've implemented myself the DQN, Double Q DQN and Dueling DQN.
The benchmarks are on going.

----------------
## Setup 


Clone the code repo and install the requirements.

```
git clone https://github.com/VCanete/DRL-Atari.git
git checkout releasev1
cd DRL-Atari
python setup.py install
pip install -r requirements.txt
```
----------------
## Run the agents

To display an agent in cartpole:

```
python play.py --env cartpole
```

To display an agent in breakout:

```
python play.py --env atari --algo rainbow
```

The recording are saved in playground/$env_name/recording

----------------
## Acknowledgements


- [@roclark](https://github.com/roclark) for [DQN](https://github.com/roclark/openai-gym-pytorch/)
- [@lilianweng](https://github.com/lilianweng) for [DQN, Double DQN, Dueling Network Architecture](https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html)
- [@fg91](https://github.com/fg91) for [DQN, Double DQN, Dueling Network Architecture](https://github.com/fg91/Deep-Q-Learning)
- [@Kaixhin](https://github.com/Kaixhin) for [Rainbow implentation](https://github.com/Kaixhin/Rainbow)
  


This project was made under the supervision of [Arthur Aubret](https://github.com/Aubret) for the AI master's degree of the University Lyon 1.

