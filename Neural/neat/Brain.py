from random import random

from neural.genetics.genes import NodeGene, ConnectionGene
from neural.genetics.genome import Genome
from neural.maths_and_data.activations import *
from neural.maths_and_data.indexed_set import IndexedSet
from neural.neat.species import Species


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
        self.max_fitness = 0

        self.reset()

        for i in range(self.clients):
            new_genome = self.create_genome()
            for j in range(10):
                new_genome.mutate()
            self.classify_genome(new_genome)

    def reset(self):

        self.all_nodes.clear()
        self.all_connectors.clear()

        for s in self.species:
            s.members.clear()

        self.species.clear()

        for i in range(self.inputs):
            n = self.get_node()

            n.set_x(0.1)
            if self.inputs > 1:
                n.set_y((i / (self.inputs - 1)) * 0.8 + 0.1)
            else:
                n.set_y(0.5)

        for o in range(self.outputs):
            n = self.get_node()

            n.set_x(0.9)
            if self.outputs > 1:
                n.set_y((o / (self.outputs - 1)) * 0.8 + 0.1)
            else:
                n.set_y(0.5)

    def create_genome(self) -> Genome:

        new_genome = Genome(sigmoid)

        new_genome.set_brain(self)
        new_genome.set_input_size(self.inputs)
        new_genome.set_output_size(self.outputs)

        for i in range(self.inputs + self.outputs):
            node = self.get_node(i + 1)
            if i < self.inputs:
                new_genome.input_nodes.append(node)
            else:
                new_genome.output_nodes.append(node)
            new_genome.nodes.append(node)

        return new_genome

    def get_connector(self, input_node: NodeGene, output_node: NodeGene):

        new_connector = ConnectionGene(input_node, output_node)
        key = (input_node.i_num, output_node.i_num)

        conn = self.all_connectors.get(key)

        if conn is not None:
            new_connector.set_inum(conn[0])
        else:
            new_connector.set_inum(len(self.all_connectors) + 1)
            self.all_connectors[key] = [new_connector.i_num, None]

        return new_connector

    def get_replace_node(self, conn: ConnectionGene) -> NodeGene:

        key = (conn.input.i_num, conn.output.i_num)

        node_id = self.all_connectors[key][1]

        if node_id is not None:
            return self.get_node(node_id)
        else:
            self.all_connectors[key][1] = len(self.all_nodes) + 1
            new_node = self.get_node()

            new_node.set_x((conn.input.x + conn.output.x) / 2)
            new_node.set_y((conn.input.y + conn.output.y) / 2 + random() * 0.3 - 0.15)

            return new_node.copy()

    def get_node(self, n_id: int = 0) -> NodeGene:

        if n_id > len(self.all_nodes) or n_id == 0:
            return self.all_nodes.addItem(NodeGene(len(self.all_nodes) + 1))
        else:
            return self.all_nodes[n_id - 1].copy()

    def classify_genome(self, g):

        if self.species:
            for s in self.species:
                if s.add_member(g):
                    return

        self.species.addItem(Species(g))

    def get_best(self):
        return max(sum([s.members for s in self.species], start=[]), key=lambda m: m.fitness)

    def evolve(self):

        self.generation += 1

        global_fitness = 0

        for s in self.species:
            s.calculate_fitness()
            global_fitness += s.fitnessSum

        self.fittest = max(self.get_best(), self.fittest,
                           key=lambda gen: gen.fitness) if self.fittest is not None else self.get_best()
        self.max_fitness = self.get_best().fitness

        if global_fitness > 0:

            new_global_fitness = 0
            surviving_species = IndexedSet()
            total_pop = 0

            for s in self.species:
                if s.can_progress() or self.fittest in s.members:
                    new_global_fitness += s.fitnessSum
                    surviving_species.addItem(s)

                    s.cull_genomes(0.25)
                    total_pop += len(s.members)
                else:
                    s.kill()

            self.species = surviving_species

            if self.species:
                for s in self.species:
                    ratio = s.fitnessSum / new_global_fitness
                    diff = self.clients - total_pop

                    no = int(round(ratio * diff))

                    for i in range(no // 3):
                        s.breed(self.weightSet.get_breed_probs())

                    for i in range(no - no // 3):
                        new = self.fittest.copy()
                        for j in range(2):
                            new.mutate()

                        self.classify_genome(new)

                    for g in s.members:
                        g.mutate()

            else:
                for i in range(self.clients):
                    g = self.fittest.copy()

                    self.classify_genome(g)

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
        self.__MUTATION_PROBS = (0.06, 0.24, 0.11, 0.19, 0.2)

        # 1: Sexual, 1: Asexual
        self.__BREED_PROBABILITIES = (0.53, 0.34)

    def get_distance_constants(self, index: int = -1):
        return self.__DISTANCE_CONST[index] if index >= 0 else self.__DISTANCE_CONST

    def get_weight_strengths(self, index: int = -1):
        return self.__WEIGHT_STRENGTHS[index] if index >= 0 else self.__WEIGHT_STRENGTHS

    def get_mutation_probs(self, index: int = -1):
        return self.__MUTATION_PROBS[index] if index >= 0 else self.__MUTATION_PROBS

    def get_breed_probs(self, index: int = -1):
        return self.__BREED_PROBABILITIES[index] if index >= 0 else self.__BREED_PROBABILITIES
