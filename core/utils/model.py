import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense_NN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_layer):
        super(Dense_NN, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


class CNN(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
        self._input_shape = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)

    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)

    def weights_init(self, module):

        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)


class Dueling_CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Dueling_CNN, self).__init__()
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - \
            adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x

    def weights_init(self, module):

        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
