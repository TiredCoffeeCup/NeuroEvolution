from pickle import load, dump, HIGHEST_PROTOCOL
from random import random

from Neural.genetics.Genes import NodeGene, ConnectionGene
from Neural.genetics.Genome import Genome
from Neural.maths_and_data.Activations import *
from Neural.maths_and_data.IndexedSet import IndexedSet
from Neural.neat.Species import Species


class Brain:

    def __init__(self, inputs: int, outputs: int, clients: int):

        self.inputs = inputs
        self.outputs = outputs

        self.species: IndexedSet[Species] = IndexedSet()
        self.all_nodes = IndexedSet()
        self.all_connectors = dict()

        self.clients = clients
        self.weightSet = WeightSet()

        self.generation = 0

        self.fittest = None

        self.reset()

        for i in range(self.clients):
            newGenome = self.createGenome()
            for i in range(10):
                newGenome.mutate()
            self.classifyGenome(newGenome)

    def reset(self):

        self.all_nodes.clear()
        self.all_connectors.clear()

        for s in self.species:
            s.members.clear()

        self.species.clear()

        for i in range(self.inputs):
            n = self.getNode()

            n.setX(0.1)
            if self.inputs > 1:
                n.setY((i / (self.inputs - 1)) * 0.8 + 0.1)
            else:
                n.setY(0.5)

        for o in range(self.outputs):
            n = self.getNode()

            n.setX(0.9)
            if self.outputs > 1:
                n.setY((o / (self.outputs - 1)) * 0.8 + 0.1)
            else:
                n.setY(0.5)

    def createGenome(self) -> Genome:

        newGenome = Genome(sigmoid)

        newGenome.setBrain(self)
        newGenome.setInputSize(self.inputs)
        newGenome.setOutputSize(self.outputs)

        for i in range(self.inputs + self.outputs):
            node = self.getNode(i + 1)
            if i < self.inputs:
                newGenome.inputNodes.append(node)
            else:
                newGenome.outputNodes.append(node)
            newGenome.nodes.append(node)

        return newGenome

    def getConnector(self, inputNode: NodeGene, outputNode: NodeGene):

        newConnector = ConnectionGene(inputNode, outputNode)
        key = (inputNode.iNum, outputNode.iNum)

        conn = self.all_connectors.get(key, None)

        if conn is not None:
            newConnector.setInum(conn[0])
        else:
            newConnector.setInum(len(self.all_connectors) + 1)
            self.all_connectors[key] = [newConnector.iNum, None]

        return newConnector

    def getReplaceNode(self, conn: ConnectionGene) -> NodeGene:

        key = (conn.input.iNum, conn.output.iNum)

        nodeId = self.all_connectors[key][1]

        if nodeId is not None:
            return self.getNode(nodeId)
        else:
            self.all_connectors[key][1] = len(self.all_nodes) + 1
            newNode = self.getNode()

            newNode.setX((conn.input.x + conn.output.x) / 2)
            newNode.setY((conn.input.y + conn.output.y) / 2 + random() * 0.3 - 0.15)

            return newNode.copy()

    def getNode(self, id: int = 0) -> NodeGene:

        if id > len(self.all_nodes) or id == 0:
            return self.all_nodes.addItem(NodeGene(len(self.all_nodes) + 1))
        else:
            return self.all_nodes[id - 1].copy()

    def classifyGenome(self, g):

        if self.species:
            for s in self.species:
                if s.addMember(g): return

        self.species.addItem(Species(g))

    def getBest(self):
        return max(sum([s.members for s in self.species], start=[]), key=lambda m: m.fitness)

    def evolve(self):

        self.generation += 1

        globalFitness = 0

        for s in self.species:
            s.calculateFitness()
            globalFitness += s.fitnessSum

        self.fittest = max(self.getBest(), self.fittest,
                           key=lambda g: g.fitness) if self.fittest is not None else self.getBest()

        if globalFitness > 0:

            newGlobalFitness = 0
            survivingSpecies = IndexedSet()
            totalPop = 0

            for s in self.species:
                if s.canProgress():
                    newGlobalFitness += s.fitnessSum
                    survivingSpecies.addItem(s)

                    s.cullGenomes(0.25)
                    totalPop += len(s.members)
                else:
                    s.kill()

            self.species = survivingSpecies

            if self.species:
                for s in self.species:
                    ratio = s.fitnessSum / newGlobalFitness
                    diff = self.clients - totalPop

                    no = int(round(ratio * diff))

                    for i in range(no // 3):
                        s.breed(self.weightSet.getBreedProbs())

                    for i in range(no - no // 3):
                        new = self.fittest.copy()
                        for j in range(2):
                            new.mutate()

                        self.classifyGenome(new)

                    for g in s.members:
                        g.mutate()

            else:
                for i in range(self.clients):
                    g = self.fittest.copy()

                    self.classifyGenome(g)

        else:

            for s in self.species:
                for g in s.members:
                    g.mutate()

    def save(self, filename):
        with open(f'{filename}.neat', mode='wb') as file:
            dump(self, file, HIGHEST_PROTOCOL)

    def saveFittest(self, filename):
        with open(f'{filename}_fittest.gen', mode='wb') as file:
            dump(self.fittest, file, HIGHEST_PROTOCOL)

    def loadFittest(self, filename):
        with open(f'{filename}_fittest.gen', mode='rb') as file:
            genome = load(file)

        for s in self.species:
            s.members.clear()
        self.species.clear()

        for i in range(self.clients):
            copy = genome.copy()
            for j in range(5):
                copy.mutate()

            self.classifyGenome(copy)

    @staticmethod
    def load(filename):
        with open(f'{filename}.neat', mode='rb') as file:
            return load(file)


class WeightSet:

    def __init__(self):
        #  0: Weight, 1: Disjoint
        self.__DISTANCE_CONST = (1, 1)

        #  0: Random, 1: Shift
        self.__WEIGHT_STRENGTHS = (0.5, 0.5)

        # 0: Node, 1: Connector, 2: Toggle, 3: Shift, 4: Random
        self.__MUTATION_PROBS = (0.06, 0.24, 0.11, 0.19, 0.2)

        # 1: Sexual, 1: Asexual
        self.__BREED_PROBABILITIES = (0.53, 0.34)

    def getDistanceConstants(self, index: int = -1):
        return self.__DISTANCE_CONST[index] if index >= 0 else self.__DISTANCE_CONST

    def getWeightStrengths(self, index: int = -1):
        return self.__WEIGHT_STRENGTHS[index] if index >= 0 else self.__WEIGHT_STRENGTHS

    def getMutationProbs(self, index: int = -1):
        return self.__MUTATION_PROBS[index] if index >= 0 else self.__MUTATION_PROBS

    def getBreedProbs(self, index: int = -1):
        return self.__BREED_PROBABILITIES[index] if index >= 0 else self.__BREED_PROBABILITIES
