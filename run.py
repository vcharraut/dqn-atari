import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense_nn(nn.Module):
	
	def __init__(self, in_dim, out_dim, hidden_layer):
		super(Dense_nn, self).__init__()
		
		self.fc1 = nn.Linear(in_dim, hidden_layer)
		self.fc2 = nn.Linear(hidden_layer, hidden_layer)
		self.fc3 = nn.Linear(hidden_layer, out_dim)
			

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		return self.fc3(x)


env = gym.make('CartPole-v1')
done = False
state = env.reset()
qvalue = Dense_nn(4, 2, 16)
qvalue.load_state_dict(torch.load('lr=0.01_hidden=16_gamma=0.99_batchsize=128_steptarget=10.pt'))
print(qvalue.state_dict())
"""
qvalue.cuda()
qvalue.eval()
for t in range(20):
    # Run one episode until termination
    r = 0
    while not done:
        env.render()

        x = torch.from_numpy(state).float()
        x = x.to(torch.device('cuda'))
        action = qvalue(x).argmax().item()

        # Get the output of env from this action
        state, reward, done, _ = env.step(action)
        r += reward

    done = False
    env.reset()
    print(t, r)

env.close()
"""