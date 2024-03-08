import random
import numpy as np
import matplotlib
import time
from numba import jit
import multiprocessing as mp
from matplotlib import pyplot as plt

np.random.seed(123)

@jit(nopython=True)
def demand(p1, p2):
    if p1 < p2:
        d = 1 - p1
    elif p1 == p2:
        d = 0.5 * (1 - p1)
    else:
        d = 0
    return d

@jit(nopython=True)
def profit(p1, p2):
    return (p1 * demand(p1, p2))


P = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]) # price array


@jit(nopython=True)
def epsilon_greedy(Qtable, epsilon, uniform, state: int, price_grid: np.ndarray) -> int: 
    N = len(price_grid)
    assert Qtable.shape[0] == N, "Qtable must have the same number of rows as there are prices in the grid"
    assert Qtable.shape[1] == N, "Qtable must have the same number of columns as there are prices in the grid"
    assert state < N, "state must be a valid index in the price grid"
    assert state >= 0, "state must be a valid index in the price grid"

    # Draw action 
    if uniform < epsilon:
        price_index = np.random.choice(N)
        #print("random")
    else:
        price_index = np.argmax(Qtable[:, state])
        #print("ikke random")
    return price_index


@jit(nopython=True)
def Qfunction(price_grid, period, delta, alpha, theta):
    # Initialize prices and Q-tables
    price_index_i = np.random.choice(len(price_grid)) 
    price_index_j = np.random.choice(len(price_grid)) 

    Qtable_i = np.zeros((len(price_grid), len(price_grid)))
    Qtable_j = np.zeros((len(price_grid), len(price_grid)))
    Qtable_temp = np.zeros((len(price_grid), len(price_grid)))

    price_lists = np.zeros((int(period+1), int(2)))
    profit_1 = np.zeros(int(period))
    profit_2 = np.zeros(int(period))

    total_profit = np.zeros(period)

    epsilons = (1 - theta)**np.arange(period + 1)
    uniforms = np.random.uniform(0, 1, (period + 1, 2))

    for t in range(1, period + 1):
        state = price_index_j # the most recent draw of player i's price
        
        # figure out who's turn it is
        if t % 2 == 0:
            # player 2 is the responder
            player_index = 1
            opponent_index = 0
        else:
            # player 1 is the responder
            player_index = 0
            opponent_index = 1

        # current period 
        decision = price_grid[price_index_i]
        state_i_responds_to = price_grid[state]
        profit_i_current_period = profit(decision, state_i_responds_to)

        # next period 
        state_next_period = price_index_i # next_period's state is today's price
        price_index_j_next_period = epsilon_greedy(Qtable_j, epsilons[t], uniforms[t, opponent_index], state=state_next_period, price_grid=price_grid)

        price_j_next_period = price_grid[price_index_j_next_period]
        price_i_next_period = price_grid[price_index_i] # unchanged price, it's not i's turn 
        profit_i_next_period = profit(price_i_next_period, price_j_next_period)

        max_Q = np.max(Qtable_i[:, price_index_j_next_period])
        continuation_value = max_Q 

        new_estimate = profit_i_current_period + delta * profit_i_next_period + delta**2 * continuation_value

        # Update
        prev_estimate = Qtable_i[price_index_i, state]

        # Update Q-table for player i
        Qtable_i[price_index_i, state] = (1 - alpha) * prev_estimate + alpha * new_estimate

        # Profit opponent 
        profit_opponent = profit(state_i_responds_to, decision)
        
        # Profit
        total_profit[t-1] = (profit_i_current_period + profit_opponent) / 2


         # Update for the next iteration: Use the simulated next period's action as the actual action for the opponent
        if t % 2 == 0:
            profit_1[t-1] = profit_opponent
            profit_2[t-1] = profit_i_current_period
        else:
            profit_1[t-1] = profit_i_current_period
            profit_2[t-1] = profit_opponent

        # Switch player for next period
        Qtable_temp = Qtable_j
        Qtable_j = Qtable_i
        Qtable_i = Qtable_temp
        price_index_j = price_index_i
        price_index_i = price_index_j_next_period
        
        # Update pricelist
        price_lists[t, player_index] = decision

    return price_lists, total_profit, profit_1, profit_2


@jit(nopython=True)
def Simulations(sim, price, period, delta, alpha, theta):
    total_profit_sim = np.zeros((sim, period))
    profit_1_sim = np.zeros((sim, period))
    profit_2_sim = np.zeros((sim, period))
    avg_profit = np.zeros(sim)
    avg_profit_1 = np.zeros(sim)
    avg_profit_2 = np.zeros(sim)

    for i in range(sim):
        _, total_profit_array, profit_1_array, profit_2_array = Qfunction(price, period, delta, alpha, theta)
        total_profit_sim[i] = total_profit_array
        profit_1_sim[i] = profit_1_array
        profit_2_sim[i] = profit_2_array
        avg_profit_1[i] = np.mean(profit_1_array[-1000:])
        avg_profit_2[i] = np.mean(profit_2_array[-1000:])
        avg_profit[i] = np.mean(total_profit_array[-1000:])

    return total_profit_sim, avg_profit_1, avg_profit_2, profit_1_sim, profit_2_sim


start_time  = time.time()

np.random.seed(123)
total_profit_plot, avg_profit_1_plot, avg_profit_2_plot, profit_total_1, profit_total_2 = Simulations(1000, P, 500000, 0.95, 0.3, 0.0000276306)

end_time = time.time()

elapsed_time = end_time - start_time

print("Time taken to run 1000 simulations:", elapsed_time, "seconds")

def delta_prof(avg_array_1, avg_array_2, sim):
    together_array = np.vstack((avg_array_1, avg_array_2))
    together_array_mean = np.mean(together_array, axis=0)
    delta_1 = np.zeros(len(together_array_mean))
    for i in range(sim):
        delta_1[i] = ((together_array_mean[i]) / (0.125))
    return delta_1

def delta_div(delta_arr):
    new_delt = np.zeros(5)
    for i in range(len(delta_arr)):
        if delta_arr[i] <=1 and delta_arr[i] > 0.9: 
        #if delta_arr[i] == 1 :
            new_delt[4]+=1
        elif delta_arr[i] <=0.9 and delta_arr[i] > 0.8:
            new_delt[3]+=1
        elif delta_arr[i] <=0.8 and delta_arr[i] > 0.7:
            new_delt[2]+=1
        elif delta_arr[i] <= 0.7 and delta_arr[i] > 0.6:
            new_delt[1]+=1
        else:
            new_delt[0] +=1
    return new_delt

delta_1 = delta_prof(avg_profit_1_plot, avg_profit_2_plot, 1000)

delta_2 = delta_div(delta_1)

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

langs = ['[0.5 : 0.6]', ']0.6 : 0.7]', ']0.7 : 0.8]', ']0.8 : 0.9]', ']0.9 : 1]']

y_pos = np.arange(len(langs))

plt.title("Distribution of $\Delta$")
# Create bars
plt.bar(y_pos, delta_2)

addlabels(langs, delta_2 )
# Create names on the x-axis
plt.xticks(y_pos, langs)
plt.xlabel("$\Delta$")
plt.ylabel("Frequency")
#make label
label = [delta_2]
# Show graphic
plt.show()

samlet_prof = total_profit_plot.mean(0)
window_size = 1000
  
i = 0
# Initialize an empty list to store moving averages
moving_averages = []
# Loop through the array t o
#consider every window of size 1000
while i < len(samlet_prof) - window_size + 1:
  
    # Calculate the average of current window
    window_average = np.sum(samlet_prof[i:i+window_size]) / window_size
      
    # Store the average of current
    # window in moving average list
    moving_averages.append(window_average)
      
    # Shift window to right by one position
    i += 1

np.random.seed(123)
plt.plot(moving_averages, label="Average profitability")
plt.xlabel('t')
plt.ylabel('Avg. profitability')
plt.ylim(0.00, 0.15)
plt.hlines(y=0.0611, xmin=0, xmax=500000, colors='red', linestyles='--')
plt.hlines(y=0.125, xmin=0, xmax=500000, colors='red', linestyles='--')
plt.show()