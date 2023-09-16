

class BaseGene:

    def __init__(self, iNum: int) -> None:
        self.iNum = iNum

    def setInum(self, value):
        self.iNum = value

    def __eq__(self, other):
        return self.iNum == other.iNum

    def __hash__(self):
        return self.iNum


class NodeGene(BaseGene):

    def __init__(self, iNum: int, relx: float = 0, rely: float = 0) -> None:
        """
        :param iNum: **Innovation Number** of the node.

        :param relx: Relative **x position** of the node in the genome. Ranges from [0, 1]

        :param rely: Relative **y position** of the node in the genome. Ranges from [0, 1]
        """
        super().__init__(iNum)

        self.__x = relx
        self.__y = rely

        self.__output = 0

    @property
    def y(self):
        return self.__y

    @property
    def x(self):
        return self.__x

    @property
    def output(self):
        return self.__output

    def setY(self, value):
        self.__y = value

    def setX(self, value):
        self.__x = value

    def addToOutput(self, value):
        self.__output += value

    def setOutput(self, value):
        self.__output = value

    def copy(self):
        return NodeGene(self.iNum, self.x, self.y)


class ConnectionGene(BaseGene):

    def __init__(self, inp: NodeGene = None, out: NodeGene = None, iNum: int = 0, weight: float = 0) -> None:
        super().__init__(iNum)

        self.__input = inp
        self.__output = out

        self.__active = True

        self.__weight = weight

    def setActivity(self, value):
        self.__active = value

    def setInput(self, value: NodeGene):
        """
        Sets the input node of the connection

        :param value:  The node which the connection is connecting from
        """
        self.__input = value

    def setWeight(self, value: float):
        """
        Sets the weight of the connection

        :param value: the new weight of the connection gene
        """
        self.__weight = value

    def setOutput(self, value: NodeGene):
        """
        Sets the output node of the connection

        :param value: The node which the connection is connected to
        """
        self.__output = value

    @property
    def weight(self):
        return self.__weight

    @property
    def output(self):
        return self.__output

    @property
    def input(self):
        return self.__input

    @property
    def active(self):
        return self.__active
