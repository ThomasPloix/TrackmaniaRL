import numpy as np
import cv2
import mss
import time
import pyautogui
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import random

# Configuration
MEMORY_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 0.00025

# Définir la région de capture (quart inférieur droit de l'écran)
region = {"top": 540, "left": 960, "width": 960, "height": 540}

# Fonction pour capturer l'écran
def capture_screen(region):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
        return screenshot

# Fonction pour prétraiter l'image
def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame / 255.0
    return frame

# Créer le modèle DQN
def create_model(input_shape, num_actions):

    model =tf.keras.models.Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model



# Fonction pour ajouter une transition à la mémoire
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Fonction pour choisir une action
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(4)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Fonction pour préparer les lots d'entraînement
def replay(batch_size):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = model.predict(state)
        if done:
            target[0][action] = reward
        else:
            t = target_model.predict(next_state)[0]
            target[0][action] = reward + GAMMA * np.amax(t)
        model.fit(state, target, epochs=1, verbose=0)

# Fonction pour effectuer une action dans le jeu
def perform_action(action):
    if action == 0:
        pyautogui.keyDown('w')  # Accélérer
        time.sleep(0.1)
        pyautogui.keyUp('w')
    elif action == 1:
        pyautogui.keyDown('s')  # Freiner
        time.sleep(0.1)
        pyautogui.keyUp('s')
    elif action == 2:
        pyautogui.keyDown('a')  # Tourner à gauche
        time.sleep(0.1)
        pyautogui.keyUp('a')
    elif action == 3:
        pyautogui.keyDown('d')  # Tourner à droite
        time.sleep(0.1)
        pyautogui.keyUp('d')

# Fonction pour calculer la récompense
def get_reward(state, next_state):
    reward =0
    # Définir des seuils pour détecter des éléments spécifiques
    track_color = np.array([0.5, 0.5, 0.5])  #  couleur de la piste gris
    obstacle_color = np.array([0.1, 0.1, 0.1])  #  couleur des obstacles blanc

    # Détection de la progression
    track_mask = cv2.inRange(state, track_color , track_color)
    next_track_mask = cv2.inRange(next_state, track_color , track_color)

    # Calculer la différence de progression
    track_progress = np.sum(next_track_mask) - np.sum(track_mask)
    reward = track_progress / 10000.0  # Normaliser la récompense

    # Détection des collisions
    obstacle_mask = cv2.inRange(state, obstacle_color - 0.1, obstacle_color + 0.1)
    next_obstacle_mask = cv2.inRange(next_state, obstacle_color - 0.1, obstacle_color + 0.1)

    # Calculer la différence de collisions
    obstacle_collision = np.sum(next_obstacle_mask) - np.sum(obstacle_mask)
    reward -= obstacle_collision / 10000.0  # Pénaliser les collisions

    # Récompense positive pour la progression
    if track_progress > 0:
        reward += 0.1

    # Pénalité pour les collisions
    if obstacle_collision > 0:
        reward -= 0.1

    return reward

# Fonction pour vérifier si la partie est terminée
def is_done(state):
    # Définir la couleur de la piste et des obstacles
    track_color = np.array([0.5, 0.5, 0.5])  # Exemple de couleur de la piste
    obstacle_color = np.array([0.1, 0.1, 0.1])  # Exemple de couleur des obstacles

    # Définir une zone autour de la voiture (par exemple, un carré de 50x50 pixels au centre de l'image)
    center_x, center_y = 42, 42  # Centre de l'image 84x84
    zone_size = 20  # Taille de la zone autour du centre
    x1, y1 = center_x - zone_size, center_y - zone_size
    x2, y2 = center_x + zone_size, center_y + zone_size

    # Extraire la zone autour de la voiture
    zone = state[y1:y2, x1:x2]

    # Calculer la moyenne de la couleur dans la zone
    mean_color = np.mean(zone, axis=(0, 1))

    # Vérifier si la voiture est sur la piste
    if np.all(mean_color < track_color - 0.1) or np.all(mean_color > track_color + 0.1):
        return True

    # Vérifier si la voiture est sur un obstacle
    if np.all(mean_color < obstacle_color - 0.1) or np.all(mean_color > obstacle_color + 0.1):
        return True

    return False


def reset_game():
    pyautogui.keyDown('r')
    time.sleep(0.1)
    pyautogui.keyUp('r')

# Boucle d'apprentissage
steps = 0

# Créer les modèles
model = create_model((84, 84, 1), 4)  # 4 actions possibles
target_model = create_model((84, 84, 1), 4)
target_model.set_weights(model.get_weights())

# Initialiser la mémoire
memory = []

for episode in range(1000):
    state = preprocess_frame(capture_screen(region))
    state = np.reshape(state, [1, 84, 84, 1])
    total_reward = 0
    reset_game()  # Réinitialiser le jeu
    done = False
    while not done:
        action = choose_action(state, EPSILON)
        perform_action(action)  # Contrôler le jeu
        next_state = preprocess_frame(capture_screen(region))
        next_state = np.reshape(next_state, [1, 84, 84, 1])
        reward = get_reward(state, next_state)  # Calculer la récompense
        done = is_done(next_state)  # Vérifier si la partie est terminée
        remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1
        if steps % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())
        if len(memory) > BATCH_SIZE:
            replay(BATCH_SIZE)
        if done:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON}")
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY