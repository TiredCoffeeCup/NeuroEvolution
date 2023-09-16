from math import ceil
from random import choices, choice

from Neural.genetics.Genome import Genome
from Neural.maths_and_data.IndexedSet import IndexedSet


class Species:
    __MAX_DISTANCE = 0.15

    def __init__(self, representative):

        self.representative = representative
        self.members = IndexedSet()

        self.members.addItem(self.representative)

        self.fitnessHistory = []
        self.maxFitnessHist = 10
        self.fitnessSum = 0

    def addMember(self, new, force: bool = False):

        if force or self.representative.distance(new) < Species.__MAX_DISTANCE:
            self.members.addItem(new)
            return True
        return False

    def breed(self, breedProbs):

        chosenOp = choices(('as', 'se'), breedProbs)

        self.members.sort(key=lambda g: g.fitness, reverse=True)

        if chosenOp == 'as' or len(self.members) < 2:
            child = self.members[0].copy()
            child.mutate()
            self.addMember(child, force=True)

        else:
            parents = self.members[:2]
            child = Genome.crossover(parents[0], choice([parents[1], parents[1].brain.fittest]))
            child.brain.classifyGenome(child)

        return child

    def calculateFitness(self):

        if not self.members: return

        self.fitnessSum = sum([g.fitness for g in self.members])

        self.fitnessHistory.append(self.fitnessSum)

        if len(self.fitnessHistory) > self.maxFitnessHist:
            self.fitnessHistory.pop(0)

    def cullGenomes(self, fraction):

        self.members.sort(key=lambda g: g.fitness, reverse=True)

        self.members = IndexedSet(self.members[: int(ceil(fraction * len(self.members)))])

        self.representative = self.members[0]

    def canProgress(self):

        num = len(self.fitnessHistory)
        return len(
            [i for i in range(
                len(self.fitnessHistory)
            ) if i > 0  and (
                    self.fitnessHistory[i] > (sum(self.fitnessHistory[:i])/i)
            )]
        ) > num//2

    def kill(self):

        self.members.clear()
        return True
