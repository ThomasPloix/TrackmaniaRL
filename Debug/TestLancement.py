import cv2
import numpy as np
import time
from Tentative.Main import TrackmaniaEnvironment


def test_visualization(env):
    """
    Test l'environnement avec visualisation en temps réel
    """
    # Crée une fenêtre pour la visualisation
    cv2.namedWindow('Trackmania Debug', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackmania Debug', 800, 600)

    try:
        # Reset l'environnement
        state = env.reset()
        print("Environnement initialisé. Démarrez Trackmania et placez la fenêtre en premier plan.")
        time.sleep(3)  # Donne du temps pour préparer le jeu

        while True:
            # Capture l'état actuel
            frame = (state.transpose(1, 2, 0) * 255).astype(np.uint8)
            debug_frame = frame.copy()

            # Estime la position
            position = env.estimate_position(state)
            if position is not None:
                # Convertit les coordonnées normalisées en pixels
                px = int(position[0] * 84)
                py = int(position[1] * 84)

                # Dessine la position détectée
                cv2.circle(debug_frame, (px, py), 3, (0, 255, 0), -1)
                cv2.putText(debug_frame, f"Pos: ({px}, {py})",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

                # Affiche la vitesse
                velocity = env.get_velocity()
                speed = np.linalg.norm(velocity)
                cv2.putText(debug_frame, f"Speed: {speed:.2f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

            # Affiche le frame de debug
            cv2.imshow('Trackmania Debug', debug_frame)

            # Test des actions basiques
            for action in range(9):  # Test chaque action
                print(f"Testing action {action}")
                next_state, reward, done, info = env.step(action)
                state = next_state

                # Affiche les informations
                print(f"Reward: {reward:.2f}")
                print(f"Info: {info}")

                # Attend un court instant pour voir l'effet de l'action
                time.sleep(0.5)

                # Vérifie si 'q' est pressé pour quitter
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            if done:
                print("Episode terminé")
                state = env.reset()

    except KeyboardInterrupt:
        print("Test interrompu par l'utilisateur")
    finally:
        env.close()
        cv2.destroyAllWindows()


def main():
    print("Initialisation de l'environnement Trackmania...")
    env = TrackmaniaEnvironment()

    print("""
    Instructions de test :
    1. Lancez Trackmania
    2. Placez la fenêtre en mode fenêtré
    3. Assurez-vous que la fenêtre est visible et active
    4. Le test va commencer dans 3 secondes
    5. Appuyez sur 'q' pour quitter
    """)

    time.sleep(3)
    test_visualization(env)


if __name__ == "__main__":
    main()