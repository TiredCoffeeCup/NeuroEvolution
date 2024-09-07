from random import choices, sample, choice, random

from neural.genetics.genes import ConnectionGene, NodeGene
from neural.maths_and_data.indexed_set import IndexedSet


class Genome:

    def __init__(self, default_act):

        self.nodes: IndexedSet[NodeGene] = IndexedSet()
        self.connections: IndexedSet[ConnectionGene] = IndexedSet()

        self.input_nodes = []
        self.output_nodes = []

        self.__fitness = 0

        self.default_activation = default_act

        self.__brain = None

        self.__input_size = 0
        self.__output_size = 0

    def mutate(self):
        """
        Mutates the genome using a set of mutations
        :return:
        """

        # 0 - Node, 1 - Connection, 2 - Toggle, 3 - RandomWeight, 4 - ShiftWeight

        choices(
            [
                self.mutate_node,
                self.mutate_connector,
                self.toggle_connection,
                self.mutate_weight_shift,
                self.mutate_weight_random
            ],
            weights=self.brain.weightSet.get_mutation_probs(),
            k=1
        )[0]()

    def mutate_node(self):

        if len(self.connections) < 1:
            return

        connector = choice(self.connections)

        new_node: NodeGene = self.nodes.addItem(self.brain.get_replace_node(connector))

        if new_node is None:
            return

        con1 = self.connections.addItem(self.brain.get_connector(connector.input, new_node))

        con1.set_weight(connector.weight)
        con1.set_activity(connector.active)

        con2 = self.connections.addItem(self.brain.get_connector(new_node, connector.output))

        con2.set_weight(1)
        con2.set_activity(connector.active)

        connector.set_activity(False)

    def mutate_connector(self):

        i = 0

        while i < 50:
            i += 1

            nodes = sample(self.nodes, k=2)

            if nodes[0].x == nodes[1].x:
                continue

            nodes.sort(key=lambda n: n.x)

            new_con = self.connections.addItem(self.brain.get_connector(*nodes))

            if new_con is not None:
                new_con.set_weight((random() * 2 - 1) * self.brain.weightSet.get_weight_strengths(0))
                return

    def toggle_connection(self):

        if len(self.connections) < 1:
            return

        conn = choice(self.connections)
        conn.set_activity(not conn.active)

    def mutate_weight_random(self):
        if len(self.connections) < 1:
            return

        conn = choice(self.connections)
        conn.set_weight((random() * 2 - 1) * self.brain.weightSet.get_weight_strengths(0))

    def mutate_weight_shift(self):
        if len(self.connections) < 1:
            return

        conn = choice(self.connections)
        conn.set_weight(conn.weight + (random() * 2 - 1) * self.brain.weightSet.get_weight_strengths(1))

    def calculate(self, inputs: list[int]):
        if len(inputs) != self.input_size:
            raise ValueError('Input size does not match available slots')

        for i, n in enumerate(self.input_nodes):
            n.set_output(inputs[i])

        node_lis = {j: [] for j in range(len(self.nodes)) if self.nodes[j] not in self.input_nodes}

        for i in range(len(self.connections)):
            node_lis[self.nodes.index(self.connections[i].output)].append(i)

        node_lis = sorted(list(node_lis.items()), key=lambda t: self.nodes[t[0]].x)

        for n, conns in node_lis:
            node = self.nodes[n]
            node.set_output(0)

            for c in conns:
                conn = self.connections[c]

                node.add_to_output(conn.input.output * conn.weight)

            node.set_output(self.default_activation(node.output))

        return [n.output for n in self.output_nodes]

    def distance(self, other):

        s_dict = {self.connections[i].i_num: i for i in range(len(self.connections))}
        o_dict = {other.connections[j].i_num: j for j in range(len(other.connections))}

        s_set = set(s_dict)
        o_set = set(o_dict)

        if not (s_set or o_set):
            return 0

        common = s_set & o_set
        uncommon = s_set ^ o_set
        weight_diff = 0

        for i in common:
            weight_diff += abs(self.connections[s_dict[i]].weight - other.connections[o_dict[i]].weight)

        weight_dist = self.brain.weightSet.get_distance_constants(0) * weight_diff / max(len(common), 20)
        disjoint_dist = self.brain.weightSet.get_distance_constants(1) * len(uncommon) / max(len(s_set), len(o_set))

        return weight_dist + disjoint_dist

    @staticmethod
    def crossover(first, second):

        child = first.brain.create_genome()

        fit_sorted = sorted([first, second], key=lambda g: g.fitness, reverse=True)

        f_dict, l_dict = tuple({g.connections[i].i_num: i for i in range(len(g.connections))} for g in fit_sorted)

        fitter_set = set(f_dict)
        lesser_set = set(l_dict)

        for i in fitter_set & lesser_set:
            selected_conn = fit_sorted[0].connections[f_dict[i]]
            inp = first.brain.get_node(selected_conn.input.i_num)
            out = first.brain.get_node(selected_conn.output.i_num)

            child.nodes.addItem(inp)
            child.nodes.addItem(out)

            conn = child.connections.addItem(first.brain.get_connector(inp, out))

            if conn:
                conn.set_weight(selected_conn.weight)
                conn.set_activity(selected_conn.active)

        for i in (fitter_set - lesser_set):
            selected_conn = fit_sorted[0].connections[f_dict[i]]
            inp = first.brain.get_node(selected_conn.input.i_num)
            out = first.brain.get_node(selected_conn.output.i_num)

            child.nodes.addItem(inp)
            child.nodes.addItem(out)
            conn = child.connections.addItem(first.brain.get_connector(inp, out))
            if conn:
                conn.set_weight(selected_conn.weight)
                conn.set_activity(selected_conn.active)

        return child

    def set_output_size(self, value):
        self.__output_size = value

    def set_input_size(self, value):
        self.__input_size = value

    def set_brain(self, value):
        self.__brain = value

    def set_fitness(self, value):
        self.__fitness = value

    def copy(self):
        clone = self.brain.create_genome()

        for conn in self.connections:
            inp = self.brain.get_node(conn.input.i_num)
            out = self.brain.get_node(conn.output.i_num)

            clone.nodes.append(inp)
            clone.nodes.append(out)

            new_conn = clone.connections.addItem(self.brain.get_connector(inp, out))

            new_conn.set_weight(conn.weight)
            new_conn.set_activity(conn.active)

        return clone

    def kill(self):

        self.nodes.clear()
        self.connections.clear()

        return True

    @property
    def input_size(self):
        return self.__input_size

    @property
    def output_size(self):
        return self.__output_size

    @property
    def brain(self):
        return self.__brain

    @property
    def fitness(self):
        return self.__fitness

    def __str__(self):
        return str([n.output for n in self.output_nodes])
