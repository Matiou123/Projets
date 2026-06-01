`transformer.py` est le fichier du modèle et de l'entraînement. Il suffit de l'éxécuter pour entrainer le transformeur sur les oeuvres de Shakespeare qui se trouve dans `input.txt`. 
Il sauvegarde aussi les hyperparamètres du modèle dans un le fichier `transformer.yaml` qui permet de facilement charger les hyperaparamètres du modèle entrainer
dans un nouveau modèle instancier, comme c'est fait dans `inférence.py`. 

Les dépendances sont seulement `Pytorch`, `yaml` et un GPU (j'ai seulement implémenté avec "cuda"). Si vous n'avez pas yaml d'installé, voici la commande pour l'installer dans votre environnement

```bash
pip install pyyaml
```
## Référence
La vidéo Andrej sur GPT https://www.youtube.com/watch?v=kCc8FmEb1nY

L'article de référence principal https://arxiv.org/pdf/1706.03762
