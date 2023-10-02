import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


class ActionValueEstimator:
    def __init__(self, env, learning_rate = 1e-3):
        assert(len(env.observation_space.shape) == 1)
        nr_features = env.observation_space.shape[0]
        assert(len(env.action_space.shape) == 0)
        nr_actions = env.action_space.n

        self._model = nn.Sequential(
            nn.Linear(nr_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, nr_actions)).to(device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, observation):
        return self._model(torch.tensor(observation).to(device)).cpu().detach().numpy()

    def train(self, observations, actions, values):
        observations = torch.tensor(observations).to(device)
        actions = torch.LongTensor(actions).to(device)
        values = torch.tensor(values).to(device)

        multi_values = self._model(observations)
        action_values = torch.gather(multi_values, 1, actions)

        loss = F.mse_loss(action_values, values)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def load_weights(self, weights_file):
        self._model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))

    def save_weights(self, weights_file):
        torch.save(self._model.state_dict(), weights_file)


class StateValueEstimator:
    def __init__(self, env, learning_rate = 1e-3):
        assert(len(env.observation_space.shape) == 1)
        nr_features = env.observation_space.shape[0]

        self._model = nn.Sequential(
            nn.Linear(nr_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)).to(device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def predict(self, observation):
        return self._model(torch.tensor(observation).to(device)).cpu().detach().numpy()

    def train(self, observations, values):
        observations = torch.tensor(observations).to(device)
        values = torch.tensor(values).to(device)

        state_values = self._model(observations)

        loss = F.mse_loss(state_values, values)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def load_weights(self, weights_file):
        self._model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))

    def save_weights(self, weights_file):
        torch.save(self._model.state_dict(), weights_file)


class PolicyEstimator:
    def __init__(self, env, learning_rate = 1e-3):
        assert(len(env.observation_space.shape) == 1)
        nr_features = env.observation_space.shape[0]
        assert(len(env.action_space.shape) == 0)
        nr_actions = env.action_space.n

        self._model = nn.Sequential(
            nn.Linear(nr_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, nr_actions)).to(device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        
    @torch.no_grad()
    def predict(self, observation):
        logits = self._model(torch.tensor(observation).to(device))
        probs = F.softmax(logits, -1)
        return probs.cpu().detach().numpy()

    def train(self, observations, actions, advantages):
        observations = torch.tensor(observations).to(device)
        actions = torch.LongTensor(actions).to(device).squeeze(-1)
        advantages = torch.tensor(advantages).to(device).squeeze(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Predict logits
        logits = self._model(observations)

        # Policy gradient
        losses = F.cross_entropy(logits, actions, reduction='none')
        loss = (losses * advantages).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def load_weights(self, weights_file):
        self._model.load_state_dict(torch.load(weights_file, map_location=torch.device(device)))

    def save_weights(self, weights_file):
        torch.save(self._model.state_dict(), weights_file)
