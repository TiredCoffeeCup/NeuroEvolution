import pygame as pg
from random import random, randint
from math import pi

WIDTH, HEIGHT = DIMS = (800, 500)

INITIAL_LIFETIME = 10
LOSS_OF_LIFE = 1

CLIENTS =  20
OUTPUT_FUNCTS = {
    0: lambda x: x.moveForward(),
    1: lambda x: x.turnLeft(),
    2: lambda x: x.turnRight()
}

SUBJECT_MSPEED = 150
SUBJECT_RSPEED = 6.28

SUBJECT_LIVE = pg.image.load('assets/subject_live.png')
SUBJECT_DEAD = pg.image.load('assets/subject_dead.png')

FOOD_AMT = 50
FOODLIFE = 2
FOOD_FREQ = 0.8
FOODSIZE = 20
FRICTION = 0.08

EVOLVING = True

BG_COLOR = (5, 170, 200)
FPS = 60