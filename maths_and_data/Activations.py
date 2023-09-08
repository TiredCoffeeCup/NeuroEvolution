from math import e, pi


def sigmoid(value) -> float:
    return 1/(1+e**(-value))