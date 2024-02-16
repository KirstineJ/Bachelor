def Qfunction(price, period, delta, alpha, theta):
    # Initialize prices and Q-tables
    price1 = int(np.random.choice(len(price))) # index i matrix kan ikke være en float
    price2 = 0

    price1_list = []
    price2_list = []

    price1_list.append(price[price1])
    price1_list.append(price[price1])
    price2_list.append(price[price2])
    price2_list.append(price[price2])

    state = 0

    Qtable_1 = np.zeros((len(price), len(price)))
    Qtable_2 = np.zeros((len(price), len(price)))

    profit_1_list = [] # kan måske slettes senere
    profit_2_list = [] # kan måske slettes senere

    avg_profit_1 = 0
    avg_profit_2 = 0

    profit_1_list.append(avg_profit_1)
    profit_1_list.append(avg_profit_1)
    profit_2_list.append(avg_profit_2)
    profit_2_list.append(avg_profit_2)


    t = 3
    i = 1
    j = 2

    for t in range(t, period + 1):
        epsilon = (1 - theta)**t
        if (t % 2) != 0:

            # Previous estimate
            prev_estimate = Qtable_1[price1, price2]

            # New estimate
            profit_current_state = Profit(price[price1], price[price2])

            profit_next_state = delta * Profit(price[price1], price[state])

            max_Q = np.argmax(Qtable_1[:, state])

            max_Q_next_state = delta**2 * max_Q

            new_estimate = profit_current_state + profit_next_state + max_Q_next_state

            # Update
            Qtable_1[price1, price2] = (1 - alpha) * prev_estimate + alpha * new_estimate

            # Set p_it
            if np.random.uniform(0,1) < epsilon:
                price1 = int(np.random.choice(len(price)))
            else:
                state = price2
                price1 = np.argmax(Qtable_1[:, state])

            state = price1

            price1_list.append(price[price1])
            price1_list.append(price[price1])
            profit_1_list.append(profit_current_state)
            profit_1_list.append(profit_current_state)

        else:
            # Previous estimate
            prev_estimate = Qtable_2[price2, state]

            # New estimate
            profit_current_state = Profit(price[price2], price[price1])

            profit_next_state = delta * Profit(price[price2], price[state])

            max_Q = np.argmax(Qtable_2[:, state])

            max_Q_next_state = delta**2 * max_Q

            new_estimate = profit_current_state + profit_next_state + max_Q_next_state

            # Update
            Qtable_2[price2, state] = (1 - alpha) * prev_estimate + alpha * new_estimate 

            # Set p_jt
            if np.random.uniform(0,1) < epsilon:
                price2 = int(np.random.choice(len(price)))
            else:
                state = price1
                price2 = np.argmax(Qtable_2[:, state])


            state = price2

            price2_list.append(price[price2])
            price2_list.append(price[price2])
            profit_2_list.append(profit_current_state)
            profit_2_list.append(profit_current_state)

        # Update t, i and j
        t = t + 1
        i = j
        j = i

    return Qtable_1, Qtable_2, price1_list, price2_list, profit_1_list, profit_2_list