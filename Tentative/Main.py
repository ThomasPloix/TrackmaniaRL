import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import mss
import pydirectinput
import win32gui
import cv2
from PIL import Image


class TrackmaniaEnvironment:
    def __init__(self):
        # Constantes pour la capture d'écran
        self.sct = mss.mss()
        self.window_name = "Trackmania"  # À ajuster selon le titre exact de la fenêtre
        self.hwnd = win32gui.FindWindow(None, self.window_name)
        if not self.hwnd:
            raise Exception("Fenêtre Trackmania non trouvée")

        # Touches pour le contrôle (codes virtuels Windows)
        # Configuration de pydirectinput
        pydirectinput.PAUSE = 0.0  # Désactive le délai entre les actions

        # Touches pour le contrôle
        self.controls = {
            'up': 'up',  # Flèche haut
            'down': 'down',  # Flèche bas
            'left': 'left',  # Flèche gauche
            'right': 'right',  # Flèche droite
            'reset': 'r'  # Touche 'R' pour reset
        }
        # Définition des actions possibles
        self.actions = [
            [],  # Rien
            ['left'],  # Gauche
            ['right'],  # Droite
            ['up'],  # Accélérer
            ['up', 'left'],  # Accélérer + Gauche
            ['up', 'right'],  # Accélérer + Droite
            ['down'],  # Freiner
            ['down', 'left'],  # Freiner + Gauche
            ['down', 'right']  # Freiner + Droite
        ]

        # Variables pour le suivi de la progression
        self.last_checkpoint_time = time.time()
        self.last_position = None
        self.frames_stuck = 0

        # Paramètres pour la détection de la voiture
        self.car_color_lower = np.array([0, 0, 200])  # Rouge clair en BGR
        self.car_color_upper = np.array([50, 50, 255])  # Rouge foncé en BGR

        # Pour le suivi de mouvement
        self.prev_gray = None
        self.tracked_points = None

        # Pour le calcul de vitesse
        self.velocity = np.array([0.0, 0.0])
        self.pressed_keys = set()

        # Paramètres pour le système de récompenses
        self.prev_positions = deque(maxlen=10)  # Historique des positions
        self.prev_speeds = deque(maxlen=5)  # Historique des vitesses
        self.last_checkpoint_time = time.time()
        self.last_speed = 0
        self.track_direction = np.array([1.0, 0.0])  # Direction générale de la piste
        self.lap_progress = 0.0  # Progression sur le tour (0 à 1)

        # Paramètres de récompenses
        self.reward_weights = {
            'speed': 2.0,  # Récompense pour la vitesse
            'progress': 5.0,  # Récompense pour la progression
            'checkpoint': 50.0,  # Bonus pour passage de checkpoint
            'stability': -1.0,  # Pénalité pour instabilité
            'crash': -100.0,  # Pénalité pour crash
            'wall_contact': -2.0,  # Pénalité pour contact avec mur
            'finish': 1000.0  # Bonus pour franchissement ligne d'arrivée
        }

    def get_window_rect(self):
        """Récupère les coordonnées de la fenêtre du jeu"""
        rect = win32gui.GetWindowRect(self.hwnd)
        return {
            'left': rect[0],
            'top': rect[1],
            'width': rect[2] - rect[0],
            'height': rect[3] - rect[1]
        }

    def capture_screen(self):
        """Capture l'écran du jeu et le prétraite"""
        # Capture la fenêtre du jeu
        window_rect = self.get_window_rect()
        screenshot = self.sct.grab(window_rect)

        # Convertit en format numpy
        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        img = np.array(img)

        # Prétraitement
        img = cv2.resize(img, (84, 84))  # Redimensionne à la taille d'entrée du réseau
        img = img.transpose(2, 0, 1)  # Change l'ordre des canaux pour PyTorch (C, H, W)
        img = img.astype(np.float32) / 255.0  # Normalisation

        return img

    def press_key(self, key):
        """Simule l'appui d'une touche"""
        if key not in self.pressed_keys:
            pydirectinput.keyDown(self.controls[key])
            self.pressed_keys.add(key)

    def release_key(self, key):
        """Simule le relâchement d'une touche"""
        if key in self.pressed_keys:
            pydirectinput.keyUp(self.controls[key])
            self.pressed_keys.remove(key)

    def release_all_keys(self):
        """Relâche toutes les touches pressées"""
        for key in list(self.pressed_keys):
            self.release_key(key)

    def execute_action(self, action_idx):
        """Exécute une action donnée"""
        # Récupère les touches pour cette action
        action_keys = self.actions[action_idx]

        # Relâche les touches qui ne sont plus nécessaires
        for key in list(self.pressed_keys):
            if key not in action_keys:
                self.release_key(key)

        # Appuie sur les nouvelles touches
        for key in action_keys:
            if key not in self.pressed_keys:
                self.press_key(key)

    def get_state(self):
        """Récupère l'état actuel du jeu"""
        return self.capture_screen()

    def calculate_reward(self, current_position, frame):
        """Calcule la récompense basée sur plusieurs facteurs"""
        try:
            reward = 0

            # 1. Vitesse
            current_speed = self.calculate_speed(current_position)
            speed_reward = self.reward_weights['speed'] * current_speed
            reward += speed_reward

            # 2. Progression sur la piste
            if len(self.prev_positions) > 0:
                # Direction du mouvement
                movement = current_position - self.prev_positions[-1]
                track_dir = self.estimate_track_direction(current_position)

                # Récompense la progression dans la bonne direction
                alignment = np.dot(movement, track_dir)
                progress_reward = self.reward_weights['progress'] * max(0, alignment)
                reward += progress_reward

            # 3. Stabilité
            if len(self.prev_speeds) >= 2:
                speed_variation = abs(current_speed - self.prev_speeds[-1])
                stability_penalty = self.reward_weights['stability'] * speed_variation
                reward += stability_penalty

            # # 4. Contact avec les murs - avec gestion des erreurs
            # if self.detect_wall_contact(frame):
            #     reward += self.reward_weights['wall_contact']

            # 5. Détection de crash ou blocage
            if self.detect_crash():
                reward += self.reward_weights['crash']

            if current_position < 0.01:  # Seuil ajusté
                self.frames_stuck += 1
                if self.frames_stuck > 30:
                    reward -= 2.0
            else:
                self.frames_stuck = 0
            # 6. Mise à jour des variables de suivi
            self.prev_positions.append(current_position)
            self.last_speed = current_speed

            return reward

        except Exception as e:
            print(f"Erreur dans calculate_reward: {e}")
            return 0.0

    def calculate_speed(self, current_position):
        """Calcule la vitesse à partir des positions précédentes"""
        if len(self.prev_positions) > 0:
            positions = list(self.prev_positions)
            positions.append(current_position)

            # Calcule la vitesse moyenne sur plusieurs frames
            speeds = [np.linalg.norm(positions[i + 1] - positions[i])
                      for i in range(len(positions) - 1)]
            current_speed = np.mean(speeds) if speeds else 0

            self.prev_speeds.append(current_speed)
            return current_speed
        return 0

    def estimate_track_direction(self, current_position):
        """Estime la direction locale de la piste"""
        if len(self.prev_positions) >= 3:
            positions = list(self.prev_positions)[-3:]
            positions.append(current_position)

            # Calcule la direction moyenne du mouvement
            directions = [positions[i + 1] - positions[i]
                          for i in range(len(positions) - 1)]
            avg_direction = np.mean(directions, axis=0)

            if np.linalg.norm(avg_direction) > 0:
                return avg_direction / np.linalg.norm(avg_direction)
        return self.track_direction

    def detect_wall_contact(self, frame):
        """Détecte si la voiture est en contact avec un mur"""
        try:
            # Assurons-nous que le frame est dans le bon format
            if isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[0] == 3:  # Si format (3, H, W)
                    frame = frame.transpose(1, 2, 0)  # Convertir en (H, W, 3)

                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
            else:
                print("Format de frame invalide")
                return False

            # Convertit l'image en HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # Définit la plage de couleurs pour les murs
            wall_lower = np.array([0, 0, 200])  # Blanc/Gris clair
            wall_upper = np.array([180, 30, 255])

            # Crée un masque pour les murs
            wall_mask = cv2.inRange(hsv, wall_lower, wall_upper)

            # Vérifie la proximité avec la position de la voiture
            car_position = self.estimate_position(frame)
            print("car position "+  car_position)
            if car_position is not None:
                x, y = int(car_position[0] * frame.shape[1]), int(car_position[1] * frame.shape[0])
                roi = wall_mask[max(0, y - 5):min(frame.shape[0], y + 5),
                      max(0, x - 5):min(frame.shape[1], x + 5)]
                return np.sum(roi) > 0
            return False

        except Exception as e:
            print(f"Erreur dans detect_wall_contact: {e}")
            return False

    def detect_crash(self):
        """Détecte si la voiture a crashé ou est bloquée"""
        return self.frames_stuck > 100  # Valeur arbitraire à ajuster

    def reset(self):
        """Réinitialise la course"""
        # Relâche d'abord toutes les touches
        self.release_all_keys()

        # Appuie sur la touche de reset
        pydirectinput.press(self.controls['reset'])

        # Réinitialise les variables de suivi
        self.last_checkpoint_time = time.time()
        self.last_position = None
        self.frames_stuck = 0

        # Attend que la course démarre
        time.sleep(1.0)

        return self.get_state()

    def step(self, action):
        """Exécute une étape de simulation avec le nouveau système de récompenses"""
        # Exécute l'action
        self.execute_action(action)

        # Attend la prochaine frame
        time.sleep(0.05)

        # Récupère le nouvel état
        next_state = self.get_state()
        frame = (next_state.transpose(1, 2, 0) * 255).astype(np.uint8)

        # Calcule la position actuelle
        current_position = self.estimate_position(next_state)

        # Calcule la récompense avec le nouveau système
        reward = self.calculate_reward(current_position, frame)

        # Vérifie si l'épisode est terminé
        done = self.detect_crash()

        # Informations supplémentaires
        info = {
            'position': current_position,
            'speed': self.last_speed,
            # 'wall_contact': self.detect_wall_contact(frame),
            'progress': self.lap_progress
        }

        return next_state, reward, done, info

    def estimate_position(self, state):
        """
        Estime la position de la voiture en utilisant une combinaison de :
        1. Détection de couleur pour trouver la voiture
        2. Flux optique pour le suivi de mouvement
        3. Filtrage pour stabiliser les estimations
        """
        # Convertit l'état en format BGR pour OpenCV
        frame = (state.transpose(1, 2, 0) * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Détection par couleur
        mask = cv2.inRange(frame_bgr, self.car_color_lower, self.car_color_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Trouve les contours de la voiture
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Prend le plus grand contour (supposé être la voiture)
            car_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(car_contour)

            if M["m00"] != 0:
                # Centre de masse du contour
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                position = np.array([cx, cy])
            else:
                position = None
        else:
            position = None

        # 2. Flux optique pour le suivi de mouvement
        if self.prev_gray is not None and position is not None:
            # Calcule le flux optique
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, frame_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )

            # Utilise le flux pour affiner la position
            flow_at_pos = flow[int(position[1]), int(position[0])]
            position = position + flow_at_pos

        # Met à jour l'image précédente
        self.prev_gray = frame_gray

        # 3. Filtrage et stabilisation
        if position is not None:
            # Ajoute la nouvelle position à l'historique
            self.prev_positions.append(position)

            if len(self.prev_positions) >= 2:
                # Calcule la vitesse
                self.velocity = position - self.prev_positions[-2]

                # Filtre les mouvements trop brusques
                if np.linalg.norm(self.velocity) > 20:  # Seuil arbitraire
                    position = self.prev_positions[-2] + self.velocity * 0.5

            # Moyenne mobile pour lisser la position
            if len(self.prev_positions) >= 3:
                position = np.mean(list(self.prev_positions)[-3:], axis=0)

            # Normalise les coordonnées entre 0 et 1
            position = position / np.array([84, 84])  # Dimensions de l'image

            return position
        else:
            # Si aucune position n'est trouvée, utilise la dernière position connue
            return self.prev_positions[-1] / np.array([84, 84]) if self.prev_positions else np.array([0.5, 0.5])

    def get_velocity(self):
        """Retourne la vitesse actuelle de la voiture"""
        return self.velocity


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
    env = TrackmaniaEnvironment()
    state_dim = (3, 84, 84)  # Exemple de dimensions pour une image RGB redimensionnée
    action_dim = 9  # Exemple : combinaisons de [rien, gauche, droite] × [rien, accélérer, freiner]

    agent = TrackmaniaAgent(state_dim, action_dim)
    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            agent.update_target_model()
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")


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
