from settings import *
from Neural import *

pg.init()

class Simulator:

    def __init__(self):
        self.display = pg.display.set_mode(DIMS)
        pg.display.set_caption('Simulator')

        self.clock = pg.time.Clock()

        self.BRAIN = Brain(1, 3, CLIENTS)
        self.VISUALIZER = Visualizer(self.BRAIN.getBest(), 200, 100)
        self.SUBJECTS = []
        self.FOOD = []

        self.dt = 0

    def pregame(self):

        self.FOOD.clear()
        self.VISUALIZER.update()

        for genome in sum([s.members for s in self.BRAIN.species], start=[]):
            self.addSubject(genome)

        while len(self.FOOD) < FOOD_AMT:
            self.addFood()
    def checkEvents(self) -> bool:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        return True

    def updateDrawContent(self):

        self.display.fill(BG_COLOR)

        for sub in self.SUBJECTS:

            if not sub.dead and len(self.FOOD):

                minFood = min(self.FOOD, key= lambda f: (f.position - sub.position).magnitude())
                minDist = (minFood.position - sub.position)

                if minDist.magnitude() < FOODSIZE//2:
                    sub.consumeFood()
                    self.addFood()
                    self.FOOD.remove(minFood)

                sub.resolveInputs(
                    [
                    # minDist.x/WIDTH,
                    # minDist.y/HEIGHT,
                    (sub.rotation.angle_to(minDist) + pi)/(2*pi)
                    ])

            if not len(self.FOOD):
                sub.kill()
                break

            if not sub.update(self.dt):
                self.SUBJECTS.remove(sub)
            sub.draw(self.display)

        for food in self.FOOD:
            food.draw(self.display)

        self.display.blit(self.VISUALIZER.surface, (20, 20))

    def addFood(self):
        offsetx = (random() * 1.8 - 0.9)
        if abs(offsetx) * random() > 0.2:
            self.FOOD.append(Food(
                pg.Vector2(
                    (WIDTH//2 + offsetx * WIDTH//2 , random() * HEIGHT)
                )))

    def addSubject(self, genome):
        self.SUBJECTS.append(
            Subject(pg.Vector2((WIDTH//2, HEIGHT//2)), genome)
        )

    def reproduceFittest(self):
        self.BRAIN.species.clear()

        for _ in range(CLIENTS):
            newGen = self.BRAIN.fittest.copy()
            newGen.mutate()

            self.BRAIN.classifyGenome(newGen)

    def mainloop(self):

        self.pregame()

        for i in self.SUBJECTS:
            print(i.genome)

        while self.checkEvents():

            if not len([sub for sub in self.SUBJECTS if not sub.dead]):
                return True

            self.updateDrawContent()

            pg.display.flip()
            self.dt = self.clock.tick(FPS)/1000

        return False
class Subject:

    def __init__(self, position: pg.Vector2, genome):

        self.position = position
        self.velocity = pg.Vector2()
        self.angularVelocity = 0
        self.rotation = pg.Vector2(0, -1)
        self.angle = 0

        self.originalImage = SUBJECT_LIVE
        self.drawingImage = self.originalImage.copy()
        self.rect = self.drawingImage.get_rect()
        self.rect.center = self.position

        self.lifetime = INITIAL_LIFETIME
        self.timeLived = 0
        self.genome = genome

        self.moved = False
        self.lastCheck = 0

        self.rotate(0)

        self.dead = False

    def moveForward(self):
        self.velocity = self.rotation*SUBJECT_MSPEED
        self.moved = True

    def turnLeft(self):
        self.angularVelocity = -SUBJECT_RSPEED

    def turnRight(self):
        self.angularVelocity = SUBJECT_RSPEED

    def rotate(self, dt):
        self.rotation = self.rotation.rotate(SUBJECT_RSPEED)
        self.angle -= SUBJECT_RSPEED
        posn = self.rect.center
        self.drawingImage = pg.transform.rotate(self.originalImage, self.angle)
        self.rect = self.drawingImage.get_rect()
        self.rect.center = posn

    def moveBackward(self):
        self.velocity = -self.rotation*SUBJECT_MSPEED

    def kill(self):
        self.genome.setFitness((self.timeLived - INITIAL_LIFETIME) * 2 // 600)
        self.originalImage = SUBJECT_DEAD

        posn = self.rect.center
        self.drawingImage = pg.transform.rotate(self.originalImage, self.angle)
        self.rect = self.drawingImage.get_rect()
        self.rect.center = posn

        self.lifetime = INITIAL_LIFETIME
        self.dead = True

    def update(self, dt):

        self.lifetime -= LOSS_OF_LIFE * dt
        self.timeLived += LOSS_OF_LIFE * dt

        if not self.dead:

            if self.velocity.magnitude():
                self.velocity -= FRICTION * self.velocity
                if self.velocity.magnitude() < 0.1: self.velocity *= 0
                self.position += self.velocity * dt
                self.rect.center = self.position

            if self.angularVelocity:
                self.rotate(dt)
                self.angularVelocity = 0

            if self.lifetime <= 0 or not(-50 < self.position.x < WIDTH+50) or not(-50 < self.position.y < HEIGHT+50):
                self.kill()

            if self.timeLived - self.lastCheck > 2:
                if not self.moved:
                    self.kill()
                else:

                    self.lastCheck = self.timeLived
                    self.moved = False



        else:

            self.lifetime -= 0.01
            if self.lifetime == 0:
                return False

            self.drawingImage.set_alpha(255 * self.lifetime/INITIAL_LIFETIME)

        return True

    def draw(self, surface: pg.surface.Surface):
        surface.blit(self.drawingImage, self.rect)

    def consumeFood(self):
        self.lifetime += FOODLIFE
    def resolveInputs(self, inputs):

        output = self.genome.calculate(inputs)

        OUTPUT_FUNCTS[output.index(max(output))](self)

class Food:

    def __init__(self, pos):
        self.position = pos
        self.image = pg.surface.Surface((FOODSIZE, FOODSIZE), pg.SRCALPHA)
        self.rect = self.image.get_rect()

        for _ in range(randint(2, 5)):

            pos = ((random()*0.4 + 0.3)*FOODSIZE, (random()*0.4 + 0.3)*FOODSIZE)
            rad = randint(int(0.1*FOODSIZE)+1,int(0.3 * FOODSIZE))

            pg.draw.circle(self.image, (200, 200, 10), pos, rad)

        mask = pg.mask.from_surface(self.image).to_surface()
        mask.set_colorkey((0, 0, 0))

        tempImage = self.image.copy()

        self.image.blit(mask, (0, 2))
        self.image.blit(mask, (0, -2))
        self.image.blit(mask, (2, 0))
        self.image.blit(mask, (-2, 0))

        self.image.blit(self.image, (0, 0), None, pg.BLEND_RGB_SUB)

        self.image.blit(tempImage, (0,0))


    def draw(self, surface):

        self.rect.center = self.position
        surface.blit(self.image, self.rect)


if __name__ == "__main__":

    sim = Simulator()

    while sim.mainloop():
        if EVOLVING:
            sim.BRAIN.evolve()
        sim.VISUALIZER.setGenome(sim.BRAIN.getBest())
