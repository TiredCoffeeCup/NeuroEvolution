from maths_and_data.IndexedSet import IndexedSet
from random import sample, choices, choice
from math import ceil
from genetics.Genome import Genome


class Species:

    __MAX_DISTANCE = 0.15

    def __init__(self, representative):

        self.representative = representative
        self.members = IndexedSet()

        self.fitnessHistory = []
        self.maxFitness = 0
        self.fitnessSum = 0

    def addMember(self, new, force: bool = False):

        if force or self.representative.distance(new) < Species.__MAX_DISTANCE:
            self.members.addItem(new)
            return True
        return False

    def breed(self, breedProbs):

        chosenOp = choices(('as', 'se'), breedProbs)

        if chosenOp == 'as':
            child = choice(self.members).copy()
            child.mutate()

        else:
            child = Genome.crossover(*sample(self.members, 2))

        self.addMember(child, force=True)
        return child

    def calculateFitness(self):

        self.fitnessSum = sum([g.fitness for g in self.members])/len(self.members)

        self.fitnessHistory.append(self.fitnessSum)
        if len(self.fitnessHistory) > self.maxFitness:
            self.fitnessHistory.pop(0)

    def cullGenomes(self, fraction):

        self.members.sort(key=lambda g: g.fitness, reverse=True)

        self.members = self.members[: int(ceil(fraction * len(self.members)))]

        self.representative = self.members[0]

    def canProgress(self):

        num = len(self.fitnessHistory)
        return sum(self.fitnessHistory)/num > self.fitnessHistory[0] or num < self.maxFitness








        



