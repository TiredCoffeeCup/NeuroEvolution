from genetics.Genes import NodeGene, ConnectionGene
from genetics.Genome import Genome
from neat.Species import Species
from maths_and_data.IndexedSet import IndexedSet
from maths_and_data.Activations import *
from random import random

class Brain:

    def __init__(self, inputs: int, outputs: int, clients: int):

        self.inputs = inputs
        self.outputs = outputs

        self.species: IndexedSet[Species] = IndexedSet()
        self.all_nodes = IndexedSet()
        self.all_connectors = dict()

        self.clients = clients

        self.globalFitness = 0

        self.weightSet = WeightSet()

        self.reset()

    def reset(self):

        self.all_nodes.clear()
        self.all_connectors.clear()

        for i in range(self.inputs):
            n = self.getNode()

            n.setX(0.1)
            n.setY((i / (self.inputs - 1)) * 0.8 + 0.1)

        for o in range(self.outputs):
            n = self.getNode()

            n.setX(0.9)
            n.setY((o / (self.outputs - 1)) * 0.8 + 0.1)

    def createGenome(self) -> Genome:

        newGenome = Genome(sigmoid)

        newGenome.setBrain(self)
        newGenome.setInputSize(self.inputs)
        newGenome.setOutputSize(self.outputs)

        for i in range(self.inputs + self.outputs):
            newGenome.nodes.append(self.getNode(i+1))

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
            newNode.setY((conn.input.y + conn.output.y) / 2 + random()*0.3 - 0.15)

            return newNode.copy()

    def getNode(self, id: int = 0) -> NodeGene:

        if id > len(self.all_nodes) or id == 0:
            return self.all_nodes.addItem(NodeGene(len(self.all_nodes) + 1))
        else:
            return self.all_nodes[id - 1].copy()

    def classifyGenomes(self, g):

        if self.species:
            for s in self.species:
                if s.addMember(g): return

        self.species.addItem(Species(g))

    def evolve(self):

        for s in self.species:
            s.calculateFitness()
            self.globalFitness += s.fitnessSum

        if self.globalFitness > 0:

            self.species = [s for s in self.species if s.canProgress()]

            totalPop = 0

            for s in self.species:
                s.cullGenomes(0.4)
                totalPop += len(s.members)

            for s in self.species:
                ratio = s.fitnessSum/self.globalFitness
                diff = self.clients - totalPop

                for i in range(int(round(ratio*diff))):
                    s.breed(self.weightSet.getBreedProbs())

                for g in s.members:
                    g.mutate()

        else:

            for s in self.species:
                for g in s.members:
                    g.mutate()

class WeightSet:

    def __init__(self):
        #  0: Weight, 1: Disjoint
        self.__DISTANCE_CONST = (1, 1)

        #  0: Random, 1: Shift
        self.__WEIGHT_STRENGTHS = (0.5, 0.5)

        # 0: Node, 1: Connector, 2: Toggle, 3: Shift, 4: Random
        self.__MUTATION_PROBS = (0.11, 0.16, 0.11, 0.19, 0.2)

        # 1: Sexual, 1: Asexual
        self.__BREED_PROBABILITIES = (0.83, 0.17)

    def getDistanceConstants(self, index: int = -1):

        return self.__DISTANCE_CONST[index] if index >= 0 else self.__DISTANCE_CONST

    def getWeightStrengths(self, index: int = -1):

        return self.__WEIGHT_STRENGTHS[index] if index >= 0 else self.__WEIGHT_STRENGTHS

    def getMutationProbs(self, index: int = -1):

        return self.__MUTATION_PROBS[index] if index >= 0 else self.__MUTATION_PROBS

    def getBreedProbs(self, index: int = -1):

        return self.__BREED_PROBABILITIES[index] if index >= 0 else self.__BREED_PROBABILITIES
