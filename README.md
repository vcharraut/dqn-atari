# DQN-Atari

The goal of this repo is to implement the DQN algorithm and try to benchmarks it in the atari environment made by OpenAI.  
Made in **Python** with the framework **PyTorch**.

**Value based:** 
- [x] DQN [[1]](#references)
- [x] Double DQN [[2]](#references)
- [x] Dueling Network Architecture [[3]](#references)

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
## Run the agents

```
python main.py --do play --env breakout --algo dueling
```

The recording are saved in playground/$env_name/recording


----------------
## Results

![Video](https://github.com/valentin-cnt/dqn-atari/blob/56ca97564fc3cfcbf49f527b748094d34bd6beae/results/openaigym.video.0.8356.video000000.mp4)

----------------
## Acknowledgements


- [@lilianweng](https://github.com/lilianweng) for [DQN, Double DQN, Dueling Network Architecture](https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html)
- [@fg91](https://github.com/fg91) for [DQN, Double DQN, Dueling Network Architecture](https://github.com/fg91/Deep-Q-Learning)
  


This project was made under the supervision of [Arthur Aubret](https://github.com/Aubret) for the AI master's degree of the University Lyon 1.


References
----------

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[3] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
