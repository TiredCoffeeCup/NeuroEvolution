import pygame as pg
from random import random, randint
from math import pi, sqrt


def angle_to(v1: pg.Vector2, v2: pg.Vector2):
    ang = abs(v2.as_polar()[1] - v1.as_polar()[1])
    if ang > 180:
        return 360 - ang

    return ang


WIDTH, HEIGHT = DIMS = (800, 500)

INITIAL_LIFETIME = 5
LOSS_OF_LIFE = 1

CLIENTS = 20
S_OUTPUT_FUNCTS = {
    0: lambda x: x.move_forward(),
    1: lambda x: x.turn_left(),
    2: lambda x: x.turn_right()
}
P_OUTPUT_FUNCTS = {
    0: lambda x: x.move_forward(),
    1: lambda x: x.turn_left(),
    2: lambda x: x.turn_right()
}

SUBJECT_MSPEED = 150
SUBJECT_RSPEED = 6.28

SUBJECT_LIVE = pg.image.load('assets/subject_live.png')
SUBJECT_DEAD = pg.image.load('assets/subject_dead.png')
PREY_IMG = pg.image.load('assets/prey.png')

PREY = 20
PREY_LIFE = 10
PREY_MSPEED = 200
PREY_RSPEED = 6.28

PREY_ENERGY_LOSS = 2
PREY_ENERGY = 10

FOOD_AMT = 50
FOODLIFE = 2
FOOD_FREQ = 0.8
FOODSIZE = 20

FRICTION = 0.08

EVOLVING = True

BG_COLOR = (5, 170, 200)
FPS = 60
