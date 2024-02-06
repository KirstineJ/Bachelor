import random
import numpy as np
import matplotlib
import time
import multiprocessing as mp
from matplotlib import pyplot as plt

def Demand(p1, p2):
    if p1 < p2:
        return (1 - p1)
    elif p1 > p2:
        return 0
    else:
        return (0.5 * (1 - p1))

def Profit(p1, p2):
    return (p1 * Demand(p1, p2))

P = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])




def Qfunction(price, period, s):
    price1 = np.random.choice(price)
    price2 = np.random.choice(price)

    Qtable_1 = np.zeros((len(price), len(price)))
    Qtable_2 = np.zeros((len(price), len(price)))

    t = 3
    i = 1
    j = 2

    for t in range(t, period + 1):
        if (t % 2) == 0:
            Qtable_1[price1, s] = 

        else:




