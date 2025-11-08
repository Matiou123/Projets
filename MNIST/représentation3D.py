import plotly.graph_objects as go
import numpy as np
import pandas as pd

def créer_prisme(x, y, z):
    # Sommets
    sommets = np.array([
        [x,     y,     0],
        [x + 1, y,     0],
        [x + 1, y + 1, 0],
        [x,     y + 1, 0],
        [x,     y,     z],
        [x + 1, y,     z],
        [x + 1, y + 1, z],
        [x,     y + 1, z]
    ])

    # Triangles 
    faces = np.array([
        [0, 1, 2], [0, 2, 3],     # Bas
        [4, 7, 6], [4, 6, 5],     # Haut
        [0, 4, 5], [0, 5, 1],     # Avant
        [1, 5, 6], [1, 6, 2],     # Droite
        [2, 6, 7], [2, 7, 3],     # Arrière
        [3, 7, 4], [3, 4, 0]      # Gauche
    ])

    x_coords, y_coords, z_coords = sommets.T
    i, j, k = faces.T

    return go.Mesh3d(
        x=x_coords, y=y_coords, z=z_coords,
        i=i, j=j, k=k,
        color='magenta',
        flatshading=True
    )

def afficher_chiffre3D(chiffre, sauve = False, titre="Exemple.html"):

    # Créer tous les prismes
    prismes = []
    for i in range(chiffre.shape[0]):
        for j in range(chiffre.shape[1]):
            if chiffre[i, j] > 0:
                prismes.append(créer_prisme(i, j, chiffre[i, j]))


    fig = go.Figure(data=prismes)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, chiffre.shape[0]], title='x'),
            yaxis=dict(range=[0, chiffre.shape[1]], title='y'),
            zaxis=dict(range=[0, 1.5], title='hauteur'),
            aspectmode="data"
        )
    )
    if sauve:
        fig.write_html(titre)
    fig.show()
