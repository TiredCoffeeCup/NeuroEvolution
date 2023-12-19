from random import choices, sample, choice, random

from Neural.genetics.Genes import ConnectionGene, NodeGene
from Neural.maths_and_data.IndexedSet import IndexedSet


class Genome:

    def __init__(self, defaultAct):

        self.nodes: IndexedSet[NodeGene] = IndexedSet()
        self.connections: IndexedSet[ConnectionGene] = IndexedSet()

        self.inputNodes = []
        self.outputNodes = []

        self.__fitness = 0

        self.defaultActivation = defaultAct

        self.__brain = None

        self.__inputSize = 0
        self.__outputSize = 0

    def mutate(self):
        """
        Mutates the genome using a set of mutations
        :return:
        """

        # 0 - Node, 1 - Connection, 2 - Toggle, 3 - RandomWeight, 4 - ShiftWeight

        choices(
            [
                self.mutateNode,
                self.mutateConnector,
                self.toggleConnection,
                self.mutateWeightShift,
                self.mutateWeightRandom
            ],
            weights=self.brain.weightSet.getMutationProbs(),
            k=1
        )[0]()

    def mutateNode(self):

        if len(self.connections) < 1: return

        connector = choice(self.connections)

        newNode: NodeGene = self.nodes.addItem(self.brain.getReplaceNode(connector))

        if newNode is None: return

        con1 = self.connections.addItem(self.brain.getConnector(connector.input, newNode))

        con1.setWeight(connector.weight)
        con1.setActivity(connector.active)

        con2 = self.connections.addItem(self.brain.getConnector(newNode, connector.output))

        con2.setWeight(1)
        con2.setActivity(connector.active)

        connector.setActivity(False)

    def mutateConnector(self):

        i = 0

        while i < 50:
            i += 1

            nodes: IndexedSet[NodeGene] = sample(self.nodes, k=2)

            if nodes[0].x == nodes[1].x: continue

            nodes.sort(key=lambda n: n.x)

            newCon = self.connections.addItem(self.brain.getConnector(*nodes))

            if newCon is not None:
                newCon.setWeight((random()*2 - 1) * self.brain.weightSet.getWeightStrengths(0))
                return

    def toggleConnection(self):

        if len(self.connections) < 1: return

        conn = choice(self.connections)
        conn.setActivity(not conn.active)

    def mutateWeightRandom(self):
        if len(self.connections) < 1: return

        conn = choice(self.connections)
        conn.setWeight((random() * 2 - 1) * self.brain.weightSet.getWeightStrengths(0))

    def mutateWeightShift(self):
        if len(self.connections) < 1: return

        conn = choice(self.connections)
        conn.setWeight(conn.weight + (random()*2 -1) * self.brain.weightSet.getWeightStrengths(1))

    def calculate(self, inputs: list[int]):
        if len(inputs) != self.inputSize: raise ValueError('Input size does not match available slots')

        for i, n in enumerate(self.inputNodes):
            n.setOutput(inputs[i])

        nodeLis = {j : [] for j in range(len(self.nodes)) if self.nodes[j] not in self.inputNodes}

        for i in range(len(self.connections)):
            nodeLis[self.nodes.index(self.connections[i].output)].append(i)

        nodeLis = sorted(list(nodeLis.items()), key=lambda t: self.nodes[t[0]].x)

        for n, conns in nodeLis:
            node =  self.nodes[n]
            node.setOutput(0)

            for c in conns:
                conn = self.connections[c]

                node.addToOutput(conn.input.output * conn.weight)

            node.setOutput(self.defaultActivation(node.output))

        return [n.output for n in self.outputNodes]

    def distance(self, other):

        sDict = {self.connections[i].iNum : i for i in range(len(self.connections))}
        oDict = {other.connections[j].iNum : j for j in range(len(other.connections))}

        s_set = set(sDict)
        o_set = set(oDict)

        if not(s_set or o_set): return 0

        common = s_set & o_set
        uncommon = s_set ^ o_set
        weightDiff = 0

        for i in common:
            weightDiff += abs(self.connections[sDict[i]].weight - other.connections[oDict[i]].weight)

        weightDist =  self.brain.weightSet.getDistanceConstants(0) * weightDiff/max(len(common), 20)
        disjointDist = self.brain.weightSet.getDistanceConstants(1) * len(uncommon)/ max(len(s_set), len(o_set))

        return weightDist + disjointDist

    @staticmethod
    def crossover(first, second):

        child = first.brain.createGenome()

        fitSorted = sorted([first, second], key=lambda g: g.fitness, reverse=True)

        fDict, lDict = tuple({g.connections[i].iNum: i for i in range(len(g.connections))} for g in fitSorted)

        fitter_set = set(fDict)
        lesser_set = set(lDict)

        for i in fitter_set & lesser_set:
            selectedConn = fitSorted[0].connections[fDict[i]]
            inp = first.brain.getNode(selectedConn.input.iNum)
            out = first.brain.getNode(selectedConn.output.iNum)

            child.nodes.addItem(inp)
            child.nodes.addItem(out)

            conn = child.connections.addItem(first.brain.getConnector(inp, out))

            if conn:
                conn.setWeight(selectedConn.weight)
                conn.setActivity(selectedConn.active)

        for i in (fitter_set - lesser_set):
            selectedConn = fitSorted[0].connections[fDict[i]]
            inp = first.brain.getNode(selectedConn.input.iNum)
            out = first.brain.getNode(selectedConn.output.iNum)

            child.nodes.addItem(inp)
            child.nodes.addItem(out)
            conn = child.connections.addItem(first.brain.getConnector(inp, out))
            if conn:
                conn.setWeight(selectedConn.weight)
                conn.setActivity(selectedConn.active)

        return child

    def setOutputSize(self, value):
        self.__outputSize = value

    def setInputSize(self, value):
        self.__inputSize = value

    def setBrain(self, value):
        self.__brain = value

    def setFitness(self, value):
        self.__fitness = value

    def copy(self):
        clone = self.brain.createGenome()

        for conn in self.connections:
            inp = self.brain.getNode(conn.input.iNum)
            out = self.brain.getNode(conn.output.iNum)

            clone.nodes.append(inp)
            clone.nodes.append(out)

            newConn = clone.connections.addItem(self.brain.getConnector(inp, out))

            newConn.setWeight(conn.weight)
            newConn.setActivity(conn.active)

        return clone

    def kill(self):

        self.nodes.clear()
        self.connections.clear()

        return True

    @property
    def inputSize(self):
        return self.__inputSize

    @property
    def outputSize(self):
        return self.__outputSize

    @property
    def brain(self):
        return self.__brain

    @property
    def fitness(self):
        return self.__fitness

    def __str__(self):
        return str([n.output for n in self.outputNodes])
