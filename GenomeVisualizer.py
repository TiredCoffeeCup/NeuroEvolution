import pygame as pg
from genetics.Genome import Genome

class Visualizer:

    def __init__(self, genome, width, height):
        if not isinstance(genome, Genome): raise ValueError('genome must be a Genome object')

        self.__surfWidth, self.__surfHeight = self.__dims = width, height
        self.__surface = pg.surface.Surface(self.dims, pg.SRCALPHA)
        self.__genome = genome

        self.update()

    @property
    def genome(self):
        return self.__genome

    @property
    def surfHeight(self):
        return self.__surfHeight

    @property
    def surface(self):
        return self.__surface

    @property
    def surfWidth(self):
        return self.__surfWidth

    @property
    def dims(self):
        return self.__dims

    def update(self):

        self.surface.fill((0, 0, 0, 0))

        for conn in self.genome.connections:
            pos1 = pg.Vector2(conn.input.x * self.surfWidth, conn.input.y * self.surfHeight)
            pos2 = pg.Vector2(conn.output.x * self.surfWidth, conn.output.y * self.surfHeight)

            color = (0, 255, 0) if conn.active else (255, 0, 0)
            pg.draw.line(self.surface, color, pos1, pos2, 2)

            wt = pg.font.Font(None, 20).render(str(conn.weight)[:5], False, color)

            self.surface.blit(wt, (pos1 + pos2)/2 - pg.Vector2(wt.get_width(), wt.get_height()) / 2 + pg.Vector2(0, 15))

        for node in self.genome.nodes:
            pos = (node.x * self.surfWidth, node.y * self.surfHeight)

            pg.draw.circle(self.surface, (0, 0, 0), pos, 0.02 * self.surfWidth)
            pg.draw.circle(self.surface, (255, 255, 255), pos, 0.01 * self.surfWidth)