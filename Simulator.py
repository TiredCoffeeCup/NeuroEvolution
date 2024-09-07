from neural import *
from settings import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import threading

mpl.rcParams['toolbar'] = 'None'

pg.init()


class Simulator:

    def __init__(self):
        self.display = pg.display.set_mode(DIMS)
        pg.display.set_caption('Simulator')

        self.clock = pg.time.Clock()

        self.SUB_BRAIN = Brain(2, 3, CLIENTS)
        self.PREY_BRAIN = Brain(2, 3, PREY)
        self.VISUALIZER_S = Visualizer(self.SUB_BRAIN.get_best(), 200, 100)
        self.VISUALIZER_P = Visualizer(self.PREY_BRAIN.get_best(), 200, 100)
        self.SUBJECTS = []
        self.FOOD = []
        self.PREY = []

        self.dt = 0

    def pregame(self):

        self.FOOD.clear()
        for i in self.PREY:
            i.kill()
        self.PREY.clear()

        fract = PREY/len(sum([s.members for s in self.PREY_BRAIN.species], start=[]))

        for s in self.PREY_BRAIN.species:
            s.cull_genomes(fract)

        self.VISUALIZER_S.update()
        self.VISUALIZER_P.update()

        for genome in sum([s.members for s in self.SUB_BRAIN.species], start=[]):
            self.add_subject(genome)

        for genome in sum([s.members for s in self.PREY_BRAIN.species], start=[]):
            while not self.add_prey(genome):
                pass

        # while len(self.FOOD) < FOOD_AMT:
        #     self.addFood()
    @staticmethod
    def check_events() -> bool:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        return True

    def update_draw_content(self):

        self.display.fill(BG_COLOR)

        for sub in self.SUBJECTS:

            if not sub.dead and len(self.PREY):

                min_food = min(self.PREY, key=lambda f: (f.position - sub.position).magnitude())
                min_dist = (min_food.position - sub.position)

                if min_dist.magnitude() < FOODSIZE // 2:
                    sub.consume_food()
                    self.add_food()
                    self.PREY.remove(min_food)

                    if random() > 0.6 and self.PREY:
                        rand_prey = self.PREY[randint(0, len(self.PREY)-1)]
                        gen = rand_prey.genome.copy()
                        self.PREY_BRAIN.classify_genome(gen)
                        for i in range(5):
                            gen.mutate()

                        self.add_prey(gen)

                sub.resolve_inputs(
                    [
                        # min_dist.x/WIDTH,
                        (250 - min_dist.magnitude())/250 if min_dist.magnitude() < 250 else 0,
                        (sub.rotation.angle_to(min_dist) + 180)/360
                    ])

            if not sub.update(self.dt):
                self.SUBJECTS.remove(sub)
            sub.draw(self.display)

        for prey in self.PREY:
            prey.draw(self.display)

            pred, visible, danger = prey.check_vision(self.SUBJECTS)
            if danger:
                clo_pred = min(danger, key=lambda f: (f.position - sub.position).magnitude())
            elif visible:
                clo_pred = min(visible, key=lambda f: (f.position - sub.position).magnitude())
            else:
                continue

            pred_dist = (clo_pred.position - prey.position)

            prey.resolve_inputs(
                [
                    (100 - pred_dist.magnitude())/100 if pred_dist.magnitude() < 100 else 0,
                    (prey.rotation.angle_to(pred_dist) + pi) / (2 * pi)
                ])

            if not prey.update(self.dt, danger, visible, pred):
                self.PREY.remove(prey)

        self.display.blit(self.VISUALIZER_S.surface, (20, 20))
        self.display.blit(self.VISUALIZER_P.surface, (WIDTH - 20 - self.VISUALIZER_P.surface.get_width(), 20))

    def add_food(self):
        offsetx = (random() * 1.8 - 0.9)
        if abs(offsetx) * random() > 0.2:
            self.FOOD.append(Food(
                pg.Vector2(
                    (WIDTH // 2 + offsetx * WIDTH // 2, random() * HEIGHT)
                )))

    def add_prey(self, genome):
        offsetx = (random() * 1.8 - 0.9)
        if abs(offsetx) * random() > 0.2:
            self.PREY.append(Prey(
                pg.Vector2(
                    (WIDTH // 2 + offsetx * WIDTH // 2, random() * HEIGHT)
                ), genome),
                )
            return True

    def add_subject(self, genome):
        self.SUBJECTS.append(
            Subject(pg.Vector2((WIDTH // 2, HEIGHT // 2)), genome)
        )

    # def reproduceFittest(self):
    #     self.SUB_BRAIN.species.clear()
    #
    #     for _ in range(CLIENTS):
    #         newGen = self.SUB_BRAIN.fittest.copy()
    #         newGen.mutate()
    #
    #         self.SUB_BRAIN.classifyGenome(newGen)

    def mainloop(self):

        self.pregame()

        while self.check_events():

            if not len([sub for sub in self.SUBJECTS if not sub.dead]):
                return True

            self.update_draw_content()

            pg.display.flip()
            self.dt = self.clock.tick(FPS) / 1000

        return False


class Being:

    def __init__(self, position: pg.Vector2, image, mv_spd, r_spd, genome):

        self.position = position
        self.velocity = pg.Vector2()
        self.angularVelocity = 0
        self.rotation = pg.Vector2(0, -1)
        self.angle = 0

        self.timeLived = 0

        self.move_speed = mv_spd
        self.rot_speed = r_spd

        self.originalImage = image
        self.drawingImage = self.originalImage.copy()
        self.rect = self.drawingImage.get_rect()
        self.rect.center = self.position

        self.genome = genome
        self.fitness = 0

        self.moved = False
        self.lastCheck = 0

        self.rotate(0)

    def move_forward(self):
        self.velocity = self.rotation * self.move_speed
        self.moved = True

    def turn_left(self):
        self.angularVelocity = -self.rot_speed

    def turn_right(self):
        self.angularVelocity = self.rot_speed

    def rotate(self, dt):
        self.rotation = self.rotation.rotate(self.rot_speed)
        self.angle -= self.rot_speed
        posn = self.rect.center
        self.drawingImage = pg.transform.rotate(self.originalImage, self.angle)
        self.rect = self.drawingImage.get_rect()
        self.rect.center = posn

    def move_backward(self):
        self.velocity = -self.rotation * self.move_speed


class Subject(Being):

    def __init__(self, position: pg.Vector2, genome):

        super().__init__(position, SUBJECT_LIVE, SUBJECT_MSPEED, SUBJECT_RSPEED, genome)

        self.lifetime = INITIAL_LIFETIME

        self.moved = False
        self.lastCheck = 0

        self.rotate(0)

        self.dead = False

    def kill(self):
        self.fitness += self.timeLived - INITIAL_LIFETIME
        self.genome.set_fitness(self.fitness)
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
                if self.velocity.magnitude() < 0.1:
                    self.velocity *= 0
                self.position += self.velocity * dt
                self.rect.center = self.position

            if self.angularVelocity:
                self.rotate(dt)
                self.angularVelocity = 0

            if self.lifetime <= 0 or not (-50 < self.position.x < WIDTH + 50) or not (
                    -50 < self.position.y < HEIGHT + 50):
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

            self.drawingImage.set_alpha(255 * self.lifetime / INITIAL_LIFETIME)

        return True

    def draw(self, surface: pg.surface.Surface):
        surface.blit(self.drawingImage, self.rect)

    def consume_food(self):
        self.fitness += 5
        self.lifetime += FOODLIFE

    def resolve_inputs(self, inputs):
        output = self.genome.calculate(inputs)

        for i in range(len(output)):
            if output[i] > 0.5:
                S_OUTPUT_FUNCTS[i](self)


class Prey(Being):
    def __init__(self, pos, genome):

        super().__init__(pos, PREY_IMG, PREY_MSPEED, PREY_RSPEED, genome)

        self.energy = PREY_ENERGY
        self.moved = False
        self.lastCheck = 0

        self.fitness = 0

        self.rotate(0)

        self.dead = False

    def kill(self):

        self.genome.set_fitness(self.fitness)
        self.dead = True

    def check_vision(self, preds):

        pred = [p for p in preds if (p.position - self.position).magnitude() < 150]
        visible = [p for p in pred if angle_to(self.rotation, (p.position - self.position)) < 80]
        danger = [p for p in pred if (p.position - self.position).magnitude() < 60]

        return pred, visible, danger

    def update(self, dt, pred, danger, vision):

        self.timeLived += LOSS_OF_LIFE * dt

        if not self.dead:

            self.fitness += 2*dt
            self.fitness -= 0.25*len(danger)*dt
            self.fitness += 0.5*(len(pred) - len(danger))*dt

            if self.velocity.magnitude():
                self.velocity -= FRICTION * self.velocity
                if self.velocity.magnitude() < 0.1:
                    self.velocity *= 0
                self.position += self.velocity * dt
                self.rect.center = self.position

                self.energy -= PREY_ENERGY_LOSS*dt

            if self.angularVelocity:
                self.rotate(dt)
                self.angularVelocity = 0

                self.energy -= PREY_ENERGY_LOSS*dt

            if self.energy <= 0 or not (-50 < self.position.x < WIDTH + 50) or not (
                    -50 < self.position.y < HEIGHT + 50):
                if self.energy <= 0:
                    self.fitness -= 20
                self.kill()
                return False

            return True

    def draw(self, surface: pg.surface.Surface):
        surface.blit(self.drawingImage, self.rect)

    def resolve_inputs(self, inputs):
        output = self.genome.calculate(inputs)

        for i in range(len(output)):
            if output[i] > 0.5:
                S_OUTPUT_FUNCTS[i](self)


class Food:

    def __init__(self, pos):
        self.position = pos
        self.image = pg.surface.Surface((FOODSIZE, FOODSIZE), pg.SRCALPHA)
        self.rect = self.image.get_rect()

        for _ in range(randint(2, 5)):
            pos = ((random() * 0.4 + 0.3) * FOODSIZE, (random() * 0.4 + 0.3) * FOODSIZE)
            rad = randint(int(0.1 * FOODSIZE) + 1, int(0.3 * FOODSIZE))

            pg.draw.circle(self.image, (200, 200, 10), pos, rad)

        mask = pg.mask.from_surface(self.image).to_surface()
        mask.set_colorkey((0, 0, 0))

        temp_image = self.image.copy()

        self.image.blit(mask, (0, 2))
        self.image.blit(mask, (0, -2))
        self.image.blit(mask, (2, 0))
        self.image.blit(mask, (-2, 0))

        self.image.blit(self.image, (0, 0), None, pg.BLEND_RGB_SUB)

        self.image.blit(temp_image, (0, 0))

    def draw(self, surface):
        self.rect.center = self.position
        surface.blit(self.image, self.rect)


def run_sim(substgts, preystgts, iterats):
    sim = Simulator()
    iters = 0
    while sim.mainloop():
        iters += 1
        if EVOLVING:
            sim.SUB_BRAIN.evolve()
            sim.PREY_BRAIN.evolve()
        sub_ftst = sim.SUB_BRAIN.max_fitness
        prey_ftst = sim.PREY_BRAIN.max_fitness
        substgts.append(sub_ftst)
        preystgts.append(prey_ftst)
        iterats.append(iters)
        sim.VISUALIZER_S.setGenome(sim.SUB_BRAIN.get_best())
        sim.VISUALIZER_P.setGenome(sim.PREY_BRAIN.get_best())


if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1)

    sub_stgths = []
    prey_stgths = []
    iterations = []

    fig.set_size_inches(3, 3)
    fig.suptitle('Max Fitness')
    def animate(i):
        print(sub_stgths, prey_stgths)
        ax.clear()
        ax.plot(iterations, sub_stgths, 'g')
        ax.plot(iterations, prey_stgths, 'y')

    thr = threading.Thread(target=run_sim, args = (sub_stgths, prey_stgths, iterations))
    thr.start()
    anim = ani.FuncAnimation(fig, animate, interval=1000)
    plt.show()