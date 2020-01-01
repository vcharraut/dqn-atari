# DRL-Atari
Deep Renforcement Learning project made with Atari game. 

- [x] DQN [[1]](#references)
- [x] Double DQN [[2]](#references)
- [ ] Prioritised Experience Replay [[3]](#references)
- [x] Dueling Network Architecture [[4]](#references)
- [ ] Multi-step Returns [[5]](#references)
- [ ] Distributional RL [[6]](#references)
- [ ] Noisy Nets [[7]](#references)

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

[1] [Human-level control through deep reinforcement
learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)  
[2] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)  
[3] [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)  
[4] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)  
[5] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[6] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)  
[7] [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295.pdf)  
