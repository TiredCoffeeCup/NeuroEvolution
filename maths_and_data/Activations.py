from math import exp, pi


def sigmoid(value) -> float:
    return 1/(1+exp(-value))