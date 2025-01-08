import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Change le backend en 'TkAgg'
import matplotlib.pyplot as plt

# Couleurs choisies
track_color = np.array([0.5, 0.5, 0.5])  # Exemple de couleur de la piste
obstacle_color = np.array([0.1, 0.1, 0.1])  # Exemple de couleur des obstacles

# Création des carrés colorés
colors = [track_color, obstacle_color]
labels = ['Track Color', 'Obstacle Color']

# Affichage des couleurs
fig, axes = plt.subplots(1, len(colors), figsize=(10, 3))
for ax, color, label in zip(axes, colors, labels):
    ax.imshow([[color]], extent=[0, 1, 0, 1])
    ax.set_title(label)
    ax.axis('off')

plt.show()
