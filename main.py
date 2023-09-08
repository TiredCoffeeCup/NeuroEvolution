from GenomeVisualizer import Visualizer
import pygame as pg
from neat.Brain import Brain


pg.init()

SCREEN = pg.display.set_mode((800, 500))
CLOCK = pg.time.Clock()

newNEAT = Brain(4, 5)

g1 = newNEAT.createGenome()
g2 = newNEAT.createGenome()

g2.setFitness(10)

mutations = 0


v1 = Visualizer(g1, 300, 150)
v2 = Visualizer(g2, 300, 150)

g_and_v = [(g1, v1, (50, 100)), (g2, v2, (450, 100))]

SCREEN.fill((255, 255, 200))
for g, v, pos in g_and_v:
    g.mutate()
    v.update()
    SCREEN.blit(v.surface, pos)

while True:

    for event in pg.event.get():

        if event.type == pg.QUIT:
            pg.quit()
            exit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_u:
                for i in g_and_v:
                    i[1].update()

            if event.key == pg.K_m:

                SCREEN.fill((255, 255, 200))
                for g, v, pos in g_and_v:
                    g.mutate()
                    v.update()
                    SCREEN.blit(v.surface, pos)
            if event.key == pg.K_c:
                print(g1.calculate([1,1,1,1]))
                print(g2.calculate([1,1,1,1]))


    mutations +=2

    # SCREEN.fill((255, 255, 200))
    # for g, v, pos in g_and_v:
    #     g.mutate()
    #     SCREEN.blit(v.surface, pos)

    pg.display.flip()
    CLOCK.tick(60)



