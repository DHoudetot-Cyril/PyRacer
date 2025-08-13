import pygame
import math

WIDTH, HEIGHT = 800, 600

car_image = pygame.Surface((40, 20), pygame.SRCALPHA)
pygame.draw.polygon(car_image, (255, 0, 0), [(0, 0), (40, 10), (0, 20)])

class Car:
    def __init__(self, x, y, speed_const=3.0):
        self.start_x = x
        self.start_y = y
        self.speed_const = speed_const
        self.reset()

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.angle = 270
        self.speed = self.speed_const
        self.alive = True

    def step(self, action):
        if not self.alive:
            return

        if action == 0:
            self.angle += 4
        elif action == 1:
            self.angle -= 4
        # action 2 = ne rien faire

        self.x += math.cos(math.radians(-self.angle)) * self.speed
        self.y += math.sin(math.radians(-self.angle)) * self.speed

    def get_state(self, walls_mask):
        sensor_angles = [-60, -30, 0, 30, 60]
        max_distance = 200
        distances = []
        for a in sensor_angles:
            dist = 0
            while dist < max_distance:
                test_x = int(self.x + math.cos(math.radians(-(self.angle + a))) * dist)
                test_y = int(self.y + math.sin(math.radians(-(self.angle + a))) * dist)
                if test_x < 0 or test_x >= WIDTH or test_y < 0 or test_y >= HEIGHT:
                    break
                if walls_mask.get_at((test_x, test_y)):
                    break
                dist += 1
            distances.append(dist / max_distance)
        return distances

    def get_mask_and_rect(self):
        rotated = pygame.transform.rotate(car_image, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        mask = pygame.mask.from_surface(rotated)
        return mask, rect

    def is_done(self, walls_mask, finish_line):
        if not self.alive:
            return True
        car_mask, car_rect = self.get_mask_and_rect()
        offset = (int(car_rect.left), int(car_rect.top))
        collision = walls_mask.overlap(car_mask, offset)
        finished = car_rect.colliderect(finish_line)
        if collision:
            self.alive = False
            return True
        if finished:
            self.alive = False
            return True
        return False
    def draw(self, surface):
        rotated = pygame.transform.rotate(car_image, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rect.topleft)
