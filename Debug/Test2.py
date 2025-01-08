import numpy as np
import cv2
import pydirectinput
import time
import win32gui
import mss
from tensorflow import keras


class TrackmaniaEnvironment:
    def __init__(self):
        self.window_name = "Trackmania"
        self.actions = [["w"], ["w", "a"], ["w", "d"], ["s"], ["a"], ["d"], []]
        self.pressed_keys = set()
        self.resset_keys = ['r']
        self.sct = mss.mss()
        self.monitor = self._get_monitor()

    def _get_monitor(self):
        hwnd = win32gui.FindWindow(None, self.window_name)
        if hwnd == 0:
            raise Exception("Trackmania window not found!")
        rect = win32gui.GetWindowRect(hwnd)
        return {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}

    def reset(self):
        self._release_all_keys()
        time.sleep(1)
        # Appuie sur la touche de reset
        pydirectinput.press(self.resset_keys[0])


        # Attend que la course démarre
        time.sleep(1.0)

        return self._capture_frame()

    def step(self, action):
        self._apply_action(action)
        time.sleep(0.1)
        frame = self._capture_frame()
        reward = self._calculate_reward(frame)
        done = self._is_done()
        return frame, reward, done

    def _capture_frame(self):
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _apply_action(self, action):
        target_keys = set(action)
        for key in self.pressed_keys - target_keys:
            pydirectinput.keyUp(key)
        for key in target_keys - self.pressed_keys:
            pydirectinput.keyDown(key)
        self.pressed_keys = target_keys

    def _release_all_keys(self):
        for key in self.pressed_keys:
            pydirectinput.keyUp(key)
        self.pressed_keys.clear()

    def _calculate_reward(self, frame):
        # Placeholder for a better reward calculation based on frame analysis
        speed = self._get_speed_from_frame(frame)
        reward = speed / 100  # Normalize speed
        return reward

    def _get_speed_from_frame(self, frame):
        # Placeholder method to extract speed from HUD
        return 50

    def _is_done(self):
        # Placeholder for crash or end detection logic
        return False


class DQNetwork:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.input_shape),
            keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse')
        return model


class TrackmaniaAgent:
    def __init__(self, environment, dq_network):
        self.environment = environment
        self.dq_network = dq_network
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(len(self.environment.actions))
        q_values = self.dq_network.model.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = np.random.choice(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.dq_network.model.predict(np.expand_dims(next_state, axis=0))[0])
            target_f = self.dq_network.model.predict(np.expand_dims(state, axis=0))[0]
            target_f[action] = target
            self.dq_network.model.fit(np.expand_dims(state, axis=0), np.expand_dims(target_f, axis=0), epochs=1,
                                      verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Example usage
environment = TrackmaniaEnvironment()
dq_network = DQNetwork(input_shape=(240, 320, 3), action_size=len(environment.actions))
agent = TrackmaniaAgent(environment, dq_network)

for episode in range(1000):
    state = environment.reset()
    for time_step in range(500):
        action = agent.act(state)
        next_state, reward, done = environment.step(environment.actions[action])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    agent.replay(32)
