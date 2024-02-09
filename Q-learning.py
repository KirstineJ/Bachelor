import random
import numpy as np
import matplotlib
import time
import multiprocessing as mp
from matplotlib import pyplot as plt

#np.random.seed(321)
np.random.seed(981)

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

    state = int(np.random.choice(len(price)))

    Qtable_1 = np.zeros((len(price), len(price)))
    Qtable_2 = np.zeros((len(price), len(price)))


    t = 3
    i = 1
    j = 2

    for t in range(t, period + 1):
        epsilon = (1 - theta)**t
        
        if (t % 2) != 0:
            print("Firm 1\n")

            # Previous estimate
            prev_estimate = Qtable_1[price1, price2]
            print("price1:\n", price1)
            print("price2:\n", price2)
            print("prev_estimate:\n", prev_estimate)

            # New estimate
            profit_current_state = Profit(price[price1], price[price2])
            print("profit_current_state:\n", profit_current_state)

            profit_next_state = delta * Profit(price[price1], price[state])
            print("state:\n", state)
            print("profit_next_state:\n", profit_next_state)

            max_Q = np.argmax(Qtable_1[:, state])
            print("Qtable_1:\n", Qtable_1[:, state])
            print("Max_Q:\n", max_Q)

            max_Q_next_state = delta**2 * max_Q
            print("max_Q_next_state:\n", max_Q_next_state)

            new_estimate = profit_current_state + profit_next_state + max_Q_next_state
            print("new_estimate:\n", new_estimate)

            # Update
            Qtable_1[price1, price2] = (1 - alpha) * prev_estimate + alpha * new_estimate
            print("Qtable_1:\n", Qtable_1)

            # Set p_it
            if np.random.uniform(0,1) < epsilon:
                price1 = int(np.random.choice(len(price)))
                print("Random\n")
            else:
                price1 = np.argmax(Qtable_1[:, state])
                print("Max\n")
            
            print("pris1:\n", price1)

        else:
            print("Firm 2\n")
            # Previous estimate
            prev_estimate = Qtable_2[price2, state]
            print("price2:\n", price2)
            print("state:\n", state)
            print("prev_estimate:\n", prev_estimate)

            # New estimate
            profit_current_state = Profit(price[price2], price[price1])
            print("price1:\n", price1)
            print("profit_current_state:\n", profit_current_state)

            profit_next_state = delta * Profit(price[price2], price[state])
            print("profit_next_state:\n", profit_next_state)

            max_Q = np.argmax(Qtable_2[:, state])
            print("Qtable_2:\n", Qtable_2[:, state])
            print("Max_Q:\n", max_Q)

            max_Q_next_state = delta**2 * max_Q
            print("max_Q_next_state:\n", max_Q_next_state)

            new_estimate = profit_current_state + profit_next_state + max_Q_next_state
            print("new_estimate:\n", new_estimate)

            # Update
            Qtable_2[price2, state] = (1 - alpha) * prev_estimate + alpha * new_estimate 
            print("Qtable_2:\n", Qtable_2)

            # Set p_jt
            if np.random.uniform(0,1) < epsilon:
                price2 = int(np.random.choice(len(price)))
                print("Random:\n")
            else:
                price2 = Qtable_2[price2, state]
                print("Max\n")
            print("pris2:\n", price2)

        # Update t, i and j
        t = t + 1
        i = j
        j = i
   
    return Qtable_1, Qtable_2, price1, price2


prev_p = np.zeros((2,2), dtype=int)
prev_p[0,0] = np.random.choice(len(P))

print(prev_p)
    
#print(Qfunction(P, 5, 0.95, 0.3, 0.00002763))

        





