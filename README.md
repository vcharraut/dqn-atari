# DRL-Atari
Deep Renforcement Learning project made with Atari game. 

- [x] DQN [[2]](#references)
- [x] Double DQN [[3]](#references)
- [ ] Prioritised Experience Replay [[4]](#references)
- [x] Dueling Network Architecture [[5]](#references)
- [ ] Multi-step Returns [[6]](#references)
- [ ] Distributional RL [[7]](#references)
- [ ] Noisy Nets [[8]](#references)

----------------
## Setup 


Clone the code repo and install the requirements.

```
git clone https://github.com/VCanete/DRL-Atari.git
cd DRL-Atari
python setup.py install
pip install -r requirements.txt
```
----------------
## Trained model

```
python play.py
```
----------------
## Train it yourself

```
python learn.py
```

----------------
## Acknowledgements


- [@roclark](https://github.com/roclark) for [DQN](https://github.com/roclark/openai-gym-pytorch/)
- [@lilianweng](https://github.com/lilianweng) for [DQN, Double DQN, Dueling Network Architecture](https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html)
- [@fg91](https://github.com/fg91) for [DQN, Double DQN, Dueling Network Architecture](https://github.com/fg91/Deep-Q-Learning)


This project was made under the supervision of [Arthur Aubret](https://github.com/Aubret) for the AI master's degree of the University Lyon 1.

----------------
## References

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[3] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[5] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[6] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[7] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
[8] [When to Use Parametric Models in Reinforcement Learning?](https://arxiv.org/abs/1906.05243)  