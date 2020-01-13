# DRL-Atari

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

# STATUS : WORK IN PROGRESS

The goal of this repo is to implement RL algorithms and try to benchmarks them in the atari environment made by OpenAI.

**Value based:** 
- [x] DQN [[1]](#references)
- [x] Double DQN [[2]](#references)
- [ ] Prioritised Experience Replay [[3]](#references)
- [x] Dueling Network Architecture [[4]](#references)
- [ ] Multi-step Returns [[5]](#references)
- [ ] Distributional RL [[6]](#references)
- [ ] Noisy Nets [[7]](#references)
- [ ] Rainbow [[8]](#references)

**Policy based:** 
- [ ] A2C [[9]](#references)
- [ ] A3C [[9]](#references)
- [ ] PPO [[10]](#references)

**Others:**
- [ ] ICM [[11]](#references)
- [ ] Deep Recurrent Q-Learning [[12]](#references)

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


References
----------

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461) 
[3] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[5] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[6] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[7] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
[8] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) 
[9] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) 
[10] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) 
[11] [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) 
[12] [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) 
