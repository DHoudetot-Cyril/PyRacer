# pyRacer - Projet de Voiture Autonome avec Deep Q-Learning (DQN)

## Description

**pyRacer** est un projet de simulation 2D de voitures autonomes développé avec **Pygame**.  
L'objectif est d'entraîner un agent (ou un groupe d'agents) à conduire une voiture sur un circuit complexe avec virages serrés, en évitant les collisions et en franchissant la ligne d’arrivée.  

L'apprentissage se fait via un réseau de neurones utilisant la méthode de **Deep Reinforcement Learning** appelée **Deep Q-Network (DQN)**.  
Le projet utilise **PyTorch**.

---
<img width="1200" height="955" alt="image" src="https://github.com/user-attachments/assets/d353b8f9-0240-42b7-be7d-d12708baaaeb" />

## Fonctionnalités principales

- Simulation graphique 2D en temps réel d’un circuit et plusieurs voitures.
- Contrôle simultané d’un batch de voitures pour un apprentissage parallèle.
- Apprentissage par renforcement avec agent DQN : actions = avancer, tourner à gauche, tourner à droite.
- Gestion des collisions via masques Pygame.
- Sauvegarde et chargement complets du modèle, replay buffer, optimiseur et hyperparamètres.
- Affichage du compteur d’épisodes et contrôles clavier pour sauvegarder (`S`) et charger (`L`).

---

## Arborescence du projet

pyRacer/  
├── car.py # Classe Car : physique, état, dessin  
├── dqn.py # Définition du réseau neuronal DQN (PyTorch)  
├── replay_memory.py # Classe ReplayMemory pour expériences  
├── main.py # Boucle principale, apprentissage DQN  
├── test_rocm.py # Test de l’installation PyTorch ROCm  
├── requirements.txt # Liste des dépendances Python  
└── README.md # Ce fichier  

---

## Méthode Deep Q-Learning (DQN)

Le DQN est une technique d’apprentissage par renforcement où un agent apprend à choisir ses actions pour maximiser la somme des récompenses futures.

### Fonctionnement général

- L’agent observe un **état** (ex: position, angle, distances aux murs).
- Il choisit une **action** (avancer, tourner à gauche, tourner à droite).
- L’environnement renvoie une **récompense** (+ si progression ou succès, - si collision).
- L’expérience `(état, action, récompense, nouvel état, done)` est stockée dans un **replay buffer**.
- Un réseau de neurones approxime la fonction Q, estimant la qualité de chaque action dans chaque état.
- Le réseau est entraîné sur des mini-batchs d’expériences prélevées aléatoirement.

### Particularités de pyRacer

- Utilisation d’un **batch de voitures** pour un apprentissage plus rapide et stable.
- Récompense positive importante pour franchir la ligne d’arrivée, pénalité forte en cas de collision.
- Récompense intermédiaire pour encourager la voiture à avancer.
- Stratégie epsilon-greedy pour équilibrer exploration/exploitation avec décroissance d’epsilon.
- Sauvegarde/chargement complet pour reprendre l’entraînement sans perte.

---

## Installation

1. Crée un environnement virtuel (optionnel mais conseillé) :

```bash
python3 -m venv venv
source venv/bin/activate
```
Installe les dépendances :


```bash
pip install -r requirements.txt
```
Lance le programme principal :

```bash
python main.py
```

### Contrôles clavier
S : sauvegarder le modèle et la mémoire d’apprentissage.

L : charger un modèle et une mémoire sauvegardés.

Fermer la fenêtre : quitter le programme.

### Contact et contributions
Ce projet est un prototype éducatif.
Les contributions, suggestions et questions sont les bienvenues via issues et pull requests.

pyRacer — Projet réalisé avec Python, Pygame et PyTorch.
