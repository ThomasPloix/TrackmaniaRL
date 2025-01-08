from tmrl import get_environment
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from collections import deque

# LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,))
# representing: (speed, 4 last LIDARs, 2 previous actions)
# actions are [gas, break, steer], analog between -1.0 and +1.0
def model(obs):
    """
    simplistic policy for LIDAR observations
    """
    deviation = obs[1].mean(0)
    deviation /= (deviation.sum() + 0.001)
    steer = 0
    for i in range(19):
        steer += (i - 9) * deviation[i]
    steer = - np.tanh(steer * 4)
    steer = min(max(steer, -1.0), 1.0)
    return np.array([1.0, 0.0, steer])

# Let us retrieve the TMRL Gymnasium environment.
# The environment you get from get_environment() depends on the content of config.json
# env = get_environment()
#
# sleep(1.0)  # just so we have time to focus the TM20 window after starting the script
#
# obs, info = env.reset()  # reset environment
# for _ in range(200):  # rtgym ensures this runs at 20Hz by default
#     act = model(obs)  # compute action
#     obs, rew, terminated, truncated, info = env.step(act)  # step (rtgym ensures healthy time-steps)
#     if terminated or truncated:
#         break
#env.wait()  # rtgym-specific method to artificially 'pause' the environment when needed

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output(input_dim), 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TrackmaniaAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_model = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=10000)

        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


def train_agent():
    env = get_environment()
    episodes = 1000
    sleep(1.0)  # just so we have time to focus the TM20 window after starting the script
    obs, info = env.reset()  # reset environment

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # action = agent.act(state)

            act = model(obs)  # compute action
            next_state, reward, done, truncated, info = env.step(act)

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        if episode % 10 == 0:

            print(f"Episode: {episode}, Total Reward: {total_reward}")
            # print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")


if __name__ == "__main__":
    try:
        print("Démarrage de l'entraînement de l'IA Trackmania...")
        print("Assurez-vous que :")
        print("1. Trackmania est lancé et visible")
        print("2. Vous êtes sur la ligne de départ")
        print("3. Le jeu est en mode plein écran ou fenêtré")

        # Attendre que l'utilisateur soit prêt
        input("Appuyez sur Entrée pour commencer l'entraînement...")

        # Démarrer l'entraînement
        train_agent()

    except KeyboardInterrupt:
        print("\nArrêt de l'entraînement...")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    finally:
        # Nettoyer l'environnement
        pass

