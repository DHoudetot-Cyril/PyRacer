import pygame
import sys
import os
import torch
import numpy as np
import random
from car import Car, WIDTH, HEIGHT
from replay_memory import ReplayMemory
from dqn import DQN

# --- Initialisation pygame et paramètres ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DQN Voiture")
clock = pygame.time.Clock()
FPS = 60
episode_count = 0
font = pygame.font.SysFont(None, 24)  # Police par défaut, taille 24

# Couleurs
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Création circuit (murs et ligne arrivée)
track_surface = pygame.Surface((WIDTH, HEIGHT))
track_surface.fill(GRAY)
walls_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# Dessin des murs (identique à ton code circuit)
pygame.draw.rect(walls_surface, BLACK, (0, 0, WIDTH, 50))         # mur haut
pygame.draw.rect(walls_surface, BLACK, (0, HEIGHT-50, WIDTH, 50)) # mur bas
pygame.draw.rect(walls_surface, BLACK, (0, 0, 50, HEIGHT))        # mur gauche
pygame.draw.rect(walls_surface, BLACK, (WIDTH-50, 0, 50, HEIGHT)) # mur droit

pygame.draw.rect(walls_surface, BLACK, (150, 0, 50, 400))          # mur vertical gauche
pygame.draw.rect(walls_surface, BLACK, (300, 200, 50, HEIGHT-200)) # mur vertical milieu
pygame.draw.rect(walls_surface, BLACK, (450, 0, 50, 400))          # mur vertical droit
pygame.draw.rect(walls_surface, BLACK, (600, 200, 50, HEIGHT-200)) # mur vertical proche arrivée

track_surface.blit(walls_surface, (0, 0))

finish_line = pygame.Rect(WIDTH - 150, 500, 100, 50)
pygame.draw.rect(track_surface, GREEN, finish_line)

walls_mask = pygame.mask.from_surface(walls_surface)

# Device torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000
LR = 1e-3

# Replay memory
memory = ReplayMemory(10000)

# DQN
dqn = DQN().to(device)
optimizer = torch.optim.Adam(dqn.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

# Multi voitures (batch)
NUM_CARS = 10
start_position = (100, 100)
start_positions = [start_position for _ in range(NUM_CARS)]
cars = [Car(x, y) for (x,y) in start_positions]

# États et récompenses
states = [car.get_state(walls_mask) for car in cars]
total_rewards = [0 for _ in range(NUM_CARS)]

steps_done = 0

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() < eps_threshold:
        return random.randrange(3)  # 3 actions possibles : avancer, tourner gauche, tourner droite
    else:
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = dqn(state_v)
            return q_values.argmax().item()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    experiences = memory.sample(BATCH_SIZE)
    batch = list(zip(*experiences))
    states_b = torch.tensor(batch[0], dtype=torch.float32).to(device)
    actions_b = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
    rewards_b = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
    next_states_b = torch.tensor(batch[3], dtype=torch.float32).to(device)
    dones_b = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

    current_q = dqn(states_b).gather(1, actions_b)
    next_q = dqn(next_states_b).max(1)[0].detach().unsqueeze(1)
    expected_q = rewards_b + GAMMA * next_q * (1 - dones_b)

    loss = criterion(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_checkpoint(path="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': dqn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'memory': memory.memory,
        'episode_count': episode_count
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(path="checkpoint.pth"):
    global episode_count
    if os.path.isfile(path):
        import replay_memory
        with torch.serialization.safe_globals([replay_memory.Experience]):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        dqn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        memory.memory = checkpoint['memory']
        episode_count = checkpoint.get('episode_count', 0)
        print(f"Checkpoint loaded from {path}")
    else:
        print(f"No checkpoint found at {path}, starting fresh")

running = True
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                save_checkpoint()
            elif event.key == pygame.K_l:
                load_checkpoint()

    # Sélection des actions pour chaque voiture
    actions = [select_action(state) for state in states]

    all_dead = True
    for i, car in enumerate(cars):
        if car.alive:
            all_dead = False
            car.step(actions[i])

            done = car.is_done(walls_mask, finish_line)
            reward = 0
            if not car.alive:  # collision
                reward = -10
            elif done:  # ligne d'arrivée franchie
                reward = 10
            else:
                reward = 0.1  # petit bonus pour avancer

            next_state = car.get_state(walls_mask)
            memory.push(states[i], actions[i], reward, next_state, done)
            states[i] = next_state
            total_rewards[i] += reward

    if all_dead:
        optimize_model()
        episode_count += 1
        print(f"Episode {episode_count} terminé, récompenses: {total_rewards}")
        # Reset toutes les voitures et les stats
        for i, car in enumerate(cars):
            car.reset()
            states[i] = car.get_state(walls_mask)
            total_rewards[i] = 0

    # Affichage Pygame
    screen.fill(WHITE)
    screen.blit(track_surface, (0, 0))
    for car in cars:
        car.draw(screen)
    text_batch = font.render(f"Episode: {episode_count}", True, (255, 255, 255))
    text_save = font.render("S: Save  L: Load", True, (255, 255, 255))
    screen.blit(text_batch, (10, 10))
    screen.blit(text_save, (10, 40))

    pygame.display.flip()

pygame.quit()
sys.exit()
