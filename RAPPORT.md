# Rapport TP DRL

**Etudiant :** Valentin Canete 11502374

## Deep Q-network sur CartPole

### Début

#### Question 1

```
import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() 
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

#### Question 2

```
list_reward, list_count_action = [], []
sum_reward, step, count_action = 0, 0, 0

sum_reward += reward

if done:
        self.env.reset()
        list_reward.append(sum_reward)
        list_count_action.append(count_action)
        sum_reward, count_action = 0, 0
        step += 1
```        

```
def log(self):
    if self.plot_reward == None:
      print('No log available')
    else:
      plt.plot(self.plot_reward, marker='D')
      plt.plot(self.plot_actions, marker='.')
      plt.show()
```

### Expérience replay

#### Question 3

```
def update_memory(self, state, action, next_state, reward, end_step):
    if len(self.memory) > self.size_memory:
      del self.memory[0]
    self.memory.append((state, action, next_state, reward, end_step))
```

#### Question 4

```
import random

def sample_batch(self, number):
    return random.sample(self.memory, number)
```

### Deep Q-learning

#### Question 5

```
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 2)
            
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
```