from game import *
from players import *
from dominion import *
from derivbot import *
from cards import *
from basic_ai import *
from rl_agent import *
import numpy as np
import pickle as pk
import tensorflow as tf
from dqltrainer import DQLagent
from dqlvaluetrainer import DQLValueAgent, ARagent
from sarsa_trainer import SarsaAgent

if 1:
###### train against random bot, deeper network, incentivize short games
# use a smaller replaybuffer so that it can learn faster
# also try to reduce epsilon sooner
# start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    #data = record_game(1000, [p1,p2],'data/1000with_action')
    data = load_game_data('data/1000with_action')

# set the network size to the input vector length
    dql = SarsaAgent(length=(data[0].shape[1]+data[1].shape[1]))
# run with a deeper network
    dql.create_model_5layers()
    dql.epsilon = 0.05
    dql.mtrain = 5000
# roughly 20 iterations of memory
    dql.replaybuffer = 2e5
    dql.fit_iterations = 10
    dql.epochs = 10
    # incentivize short games
    dql.reward_points_per_turn = 1

# first training step
    dql.add_data(data)
    dql.do_train()

# use dql vs. random player's game log to train
    for i in range(1000):
        if i%100 == 0 and i !=0 :
            dql.epsilon = 0.05/(i/10) 
            print('dql epsilon: {:.04f}'.format(dql.epsilon))
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/rand_v1_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data(500)
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.epsilon=0.0
        p_smith = smithyComboBotFactory()
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, p_smith],10))

if 0:
###### train against rlbot itself/random bot/smithy bot as opponent, deeper network, incentivize short games
# this only needs 10ish iterations to have a good fighting chance against smithyComboBot()
# use a smaller replaybuffer so that it can learn faster
# also try to reduce epsilon sooner
# start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    #data = record_game(1000, [p1,p2])
    data = load_game_data('data/1000with_action')

# set the network size to the input vector length
    dql = SarsaAgent(length=(data[0].shape[1]+data[1].shape[1]))
# run with a deeper network
    dql.create_model_5layers()
    dql.epsilon = 0.05
    dql.mtrain = 5000
# one iteration creates roughly 1e4 samples
# therefore this remembers the data of pass 20 iterations.
    dql.replaybuffer = 2e5
    dql.fit_iterations = 10
    dql.epochs = 10
    # incentivize short games
    dql.reward_points_per_turn = 1
    # I think having win reward makes it too noisy
    dql.win_reward = 0

# first training step
    dql.add_data(data)
    dql.do_train()

# use dql vs. random player's game log to train
    for i in range(1000):
        if i%100 == 0 and i !=0 :
            dql.epsilon = 0.05/(i/10) 
            print('dql epsilon: {:.04f}'.format(dql.epsilon))
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/combination_v1_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data_smithy(100)
        dql.generate_data(100)
        dql.generate_data_rl(100)
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.epsilon=0.0
        p_smith = smithyComboBotFactory()
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, p_smith],10))


if 0:
###### train against rlbot itself/random bot/smithy bot as opponent, deeper network, incentivize short games
# this only needs 10ish iterations to have a good fighting chance against smithyComboBot()
# start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    #data = record_game(1000, [p1,p2])
    data = load_game_data('data/1000with_action')

# set the network size to the input vector length
    dql = SarsaAgent(length=(data[0].shape[1]+data[1].shape[1]))
# run with a deeper network
    dql.create_model_5layers()
    dql.epsilon = 0.05
    dql.mtrain = 5000
    dql.replaybuffer = 1e6
    dql.fit_iterations = 10
    dql.epochs = 10
    # incentivize short games
    dql.reward_points_per_turn = 1
    # I think having win reward makes it too noisy
    dql.win_reward = 0

# first training step
    dql.add_data(data)
    dql.do_train()

# use dql vs. random player's game log to train
    for i in range(1000):
        if i%100 == 0 and i !=0 :
            dql.epsilon = 0.05/(i/50) 
            print('dql epsilon: {:.04f}'.format(dql.epsilon))
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/combination_v0_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data_smithy(100)
        dql.generate_data(100)
        dql.generate_data_rl(100)
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.epsilon=0.0
        p_smith = smithyComboBotFactory()
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, p_smith],10))


if 0:
###### train against smithy bot, deeper network, incentivize short games
# start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = smithyComboBotFactory()
    p2.record_history = 0
    data = record_game(1000, [p1,p2])

# set the network size to the input vector length
    dql = SarsaAgent(length=(data[0].shape[1]+data[1].shape[1]))
# run with a deeper network
    dql.create_model_5layers()
    dql.epsilon = 0.05
    dql.mtrain = 5000
    dql.replaybuffer = 1e6
    dql.fit_iterations = 10
    dql.epochs = 10
    # incentivize short games
    dql.reward_points_per_turn = 1
    dql.win_reward = 100

# first training step
    dql.add_data(data)
    dql.do_train()

# use dql vs. random player's game log to train
    for i in range(1000):
        if i%100 == 0 and i !=0 :
            dql.epsilon = 0.05/(i/50) 
            print('dql epsilon: {:.04f}'.format(dql.epsilon))
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/smithy_v0_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data_smithy(500)
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.epsilon=0.0
        p_smith = smithyComboBotFactory()
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, p_smith],10))



if 0:
###### train against random bot, deeper network, incentivize short games
# start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    #data = record_game(1000, [p1,p2],'data/1000with_action')
    data = load_game_data('data/1000with_action')

# set the network size to the input vector length
    dql = SarsaAgent(length=(data[0].shape[1]+data[1].shape[1]))
# run with a deeper network
    dql.create_model_5layers()
    dql.epsilon = 0.05
    dql.mtrain = 5000
    dql.replaybuffer = 1e6
    dql.fit_iterations = 10
    dql.epochs = 10
    # incentivize short games
    dql.reward_points_per_turn = 1

# first training step
    dql.add_data(data)
    dql.do_train()

# use dql vs. random player's game log to train
    for i in range(1000):
        if i==100:
            dql.epsilon = 0.04
            print('dql epsilon: 0.04')
        elif i==200:
            dql.epsilon = 0.03
            print('dql epsilon: 0.04')
        elif i==300:
            dql.epsilon = 0.02
            print('dql epsilon: 0.02')
        elif i==400:
            dql.epsilon = 0.01
            print('dql epsilon: 0.01')
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/rand_v0_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data(500)
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.epsilon=0.0
        p_smith = SmithyBot()
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, p_smith],10))
        



if 0:
########### continue with the RL vs. RL bot
# start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    data = record_game(1, [p1,p2])

# set the network size to the input vector length
    dql = ARagent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.fit(data) # to establish the network
    dql.load_model('model/rl_iteration_342')
    dql.epsilon = 0.01
    dql.generate_data_rl(2000)
    dql.add_data(data)
    dql.mtrain = 5000
    dql.replaybuffer = 1e6
    dql.fit_iterations = 10
    dql.epochs = 10

# use dql vs. random player's game log to train
    for i in range(1000):
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/rl_v1_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data_rl(200)
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.epsilon=0.0
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, SmithyBot()],10))
        
