import gym
import random
import numpy as np
import sys

# create enviroment
env = gym.make('FrozenLake-v0')
# crate Q-table
action_size = env.action_space.n
state_size = env.observation_space.n
Q_learning_table = np.zeros([state_size, action_size])
playingRewards0 = 0

for game in range(10000):
    state = env.reset()
    step = 0
    gameover = False
    # print("****************************************************")
    # print("EPISODE ", game)

    for step in range(99):
        # Take the action (index) that have the maximum expected future reward given that state
        action = env.action_space.sample()

        newState, reward, gameover, info = env.step(action)

        if gameover:
            playingRewards0 += reward
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)

            # We print the number of step it took.
            # print("Number of steps", step)
            # break
        state = newState

print(" ")
print("playing(Before learning) success: %d / %d" %(playingRewards0,10000))




def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben




# parameters
games = 50000  # total number of games
alpha = 0.1  # learning rate 0.1 is with optimal with 62% success.
gamma = 0.95  # discount rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

# summery variables
total = 0
winCounter = 0
result = []
rewards = []

print("Learning process: ")

for i in range(games):

    progress(i,games,"")

    state = env.reset()
    gameover = False

    while not gameover:  # until game over, or max number of steps
        # env.render()
        rand = random.random()

        # choice --> exploration
        # at first we need more exploration
        if rand < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_learning_table[state, :])

        newState, reward, gameover, prob = env.step(action)


        Q_learning_table[state, action] = Q_learning_table[state, action] + alpha * (
                reward + gamma * np.max(Q_learning_table[newState, :]) - Q_learning_table[state, action])

        state = newState
        winCounter += reward


    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)
    rewards.append(reward)

    # env.render()

if ((i + 1) % 10000 == 0):
    result.append(winCounter)
    total = total + winCounter
    winCounter = 0

print(" ")
print("////SUMMERY////")

print("wins: %d / %d" % (total, games))
print("Learning success: " + str(sum(rewards) / games))

print(" ")
print(" ")

print(Q_learning_table)


playingRewards = 0
playGameNum = 10000


for game in range(playGameNum):
    state = env.reset()
    step = 0
    gameover = False
    # print("****************************************************")
    # print("EPISODE ", game)

    for step in range(99):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Q_learning_table[state, :])

        newState, reward, gameover, info = env.step(action)

        if gameover:
            playingRewards += reward
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)

            # We print the number of step it took.
            # print("Number of steps", step)
            # break
        state = newState

print(" ")
print("playing success: %d / %d" %(playingRewards,playGameNum))

# env.close()


