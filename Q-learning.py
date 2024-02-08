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




def Qfunction(price, period, delta, alpha, theta):
    # Initialize prices and Q-tables
    price1 = int(np.random.choice(len(price))) # index i matrix kan ikke være en float
    price2 = int(np.random.choice(len(price)))

    price_s = int(np.random.choice(len(price)))

    Qtable_1 = np.zeros((len(price), len(price)))
    Qtable_2 = np.zeros((len(price), len(price)))

    epsilon = (1 - theta)**period


    t = 3
    i = 1
    j = 2

    for t in range(t, period + 1):
        if (t % 2) == 0:
            # Previous estimate
            prev_estimate = Qtable_1[price1, price_s]

            # New estimate
            profit_current_state = Profit(price1, price2)
            profit_next_state = delta * Profit(price1, price_s)

            max_Q = np.argmax(Qtable_1[:, price_s])
            max_Q_next_state = delta**2 * max_Q

            new_estimate = profit_current_state + profit_next_state + max_Q_next_state

            # Update
            Qtable_1[price1, price2] = (1 - alpha) * prev_estimate + alpha * new_estimate

        else:
            # Previous estimate
            prev_estimate = Qtable_2[price2, price_s]

            # New estimate
            profit_current_state = Profit(price2, price1)
            profit_next_state = delta * Profit(price2, price_s)

            max_Q = np.argmax(Qtable_2[:, price_s])
            max_Q_next_state = delta**2 * max_Q

            new_estimate = profit_current_state + profit_next_state + max_Q_next_state

            # Update
            Qtable_2[price2, price1] = (1 - alpha) * prev_estimate + alpha * new_estimate # måske skal priserne byttes om
        
        # Set p_it and p_jt
        if np.random.uniform(0,1) < epsilon:
            price1 = int(np.random.choice(len(price)))
        else:
            price1 = np.argmax(Qtable_1[:, price_s])

        if np.random.uniform(0,1) < epsilon:
            price2 = int(np.random.choice(len(price)))
        else:
            price2 = Qtable_2[price2, price_s]
        

        # Update t, i and j
        t = t + 1
        i = j
        j = i
   
    return Qtable_1, Qtable_2, price1, price2

    
    
print(Qfunction(P, 100, 0.95, 0.3, 0.00002763))

        





