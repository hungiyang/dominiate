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

# TODO:
# 1. add the average score of rl bots when generating data.
# 2. This gives a measure of how well they are improving 
#    Even when the win rate is 0 or 100%
do_ar_rl = 1
do_ar = 0
do_ar_smithy = 0
do_ar_fullcards = 0
do_random = 0
do_smithy = 0
do_rl = 0
do_continue = 0

# let the 


# use the full sets of cards
# use rl bot vs. rl bot to train
# add win lose reward (modified dominion.py)
if do_ar_rl:
    # start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    data = record_game(1, [p1,p2])

    # set the network size to the input vector length
    dql = ARagent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.fit(data) # to establish the network
    dql.load_model('model/full_iteration_018')
    dql.generate_data_rl(500)
    dql.add_data(data)
    dql.mtrain = 5000
    dql.replaybuffer = 1e6
    dql.fit_iterations = 10
    dql.epochs = 10
    dql.epsilon = 0.05

    # use dql vs. random player's game log to train
    for i in range(1000):
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/rl_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data_rl(200)
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.epsilon=0.0
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, SmithyBot()],10))



# use the full sets of cards
if do_ar_fullcards:
    # start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    data = record_game(1000, [p1,p2], 'data/1000full')

    # set the network size to the input vector length
    dql = ARagent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.add_data(data)
    dql.mtrain = 5000
    dql.replaybuffer = 5e6
    dql.fit_iterations = 10
    dql.epochs = 10
    dql.epsilon = 0.05

    # use dql vs. random player's game log to train
    for i in range(1000):
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_train()
        dql.save_model('./model/full_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data(100)

# train with aggregated_rewards
if do_ar:
    # start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    data = record_game(1000, [p1,p2], 'data/1000new')

    # set the network size to the input vector length
    dql = ARagent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.add_data(data)
    dql.mtrain = 5000
    dql.replaybuffer = 5e6
    dql.fit_iterations = 10
    dql.epochs = 10
    dql.epsilon = 0.05

    # use dql vs. random player's game log to train
    for i in range(1000):
        dql.do_train()
        dql.save_model('./model/v0_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data(100)

# train with aggregated_rewards
if do_ar_smithy:
    # start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = SmithyBot()
    p2.record_history = 0
    data = record_game(1000, [p1,p2])

    # set the network size to the input vector length
    dql = ARagent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.add_data(data)
    dql.mtrain = 5000
    dql.replaybuffer = 5e6
    dql.fit_iterations = 10
    dql.epochs = 10
    dql.epsilon = 0.05

    # use dql vs. random player's game log to train
    for i in range(1000):
        dql.do_train()
        dql.save_model('./model/smithy_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data_smithy(100)


# train against random bot, continue from a model with a solid win rate against random bot
if do_continue:
    # start with random player's game log
    data = load_game_data('data/5000random')
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    # data = record_game(1000, [p1,p2])

    # set the network size to the input vector length
    dql = DQLValueAgent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.add_data(data)
    dql.fit_target(dql.draw_sample())
    dql.load_model('model/v1_iteration_040')
    dql.mtrain = 1000
    dql.replaybuffer = 5e6
    dql.target_iterations=5
    dql.predict_iterations=10
    dql.epochs = 10
    dql.epsilon = 0.3
    # generate data with this RL model
    dql.generate_data(100, 'data/v2_iteration_000')


    # use dql vs. random player's game log to train
    for i in range(1000):
        dql.do_target_iteration()
        dql.save_model('./model/v2_iteration_{:03d}'.format(i+1))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data(100, 'data/v2_iteration_{:03d}'.format(i+1))



# train against random bot
if do_random:
    # start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    # data = record_game(1000, [p1,p2], 'data/iteration_0')
    # load 5000 games of presaved random
    data = load_game_data('data/5000random')

    # set the network size to the input vector length
    dql = DQLValueAgent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.add_data(data)
    dql.mtrain = 1000
    dql.replaybuffer = 5e6
    dql.target_iterations=1
    dql.predict_iterations=50
    dql.epochs = 10
    dql.epsilon = 0.1

    # use dql vs. random player's game log to train
    for i in range(1000):
        print('data generation iteration {:d}'.format(i))
        dql.do_target_iteration()
        dql.save_model('./model/v2_iteration_{:03d}'.format(i+1))
        # dql.generate_data(100, 'data/v0_iteration_{:03d}'.format(i+1))
        dql.generate_data(100)

# train against smithy bot
if do_smithy:
    # start with random player's game log
    p1 = SmithyBot()
    p1.record_history = 0
    p2 = RandomPlayer()
    p2.record_history = 1
    data = record_game(1000, [p1,p2], 'data/iteration_0')

    # set the network size to the input vector length
    dql = DQLagent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.add_data(data)
    dql.epsilon = 0.1
    dql.gamma = 0.99
    dql.mtrain = 1000
    dql.target_iterations=10
    dql.predict_iterations=100

    # use dql vs. random player's game log to train
    for i in range(1000):
        print('data generation iteration {:d}'.format(i))
        dql.do_target_iteration()
        dql.save_model('./model_smithy/iteration_{:03d}'.format(i+1))
        dql.generate_data_smithy(100, 'data_smithy/iteration_{:03d}'.format(i+1))
        vf = lambda x: dql.model_predict.predict(x)
        p1 = RLPlayer(vf, 0)
        p2 = RandomPlayer()
        print(compare_bots([p1, p2],50))


# train against itself, rlagent
if do_rl:
    # start with random player's game log
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    data = record_game(1000, [p1,p2], 'data/iteration_0')

    # set the network size to the input vector length
    dql = DQLagent(length=(data[0].shape[1]+data[1].shape[1]))
    dql.add_data(data)
    dql.replaybuffer = 1e7
    dql.epsilon = 0.05
    dql.gamma = 0.99
    dql.mtrain = 1000
    dql.target_iterations=5
    dql.predict_iterations=10

    # use dql vs. random player's game log to train
    for i in range(1000):
        print('data generation iteration {:d}'.format(i))
        dql.do_target_iteration()
        dql.save_model('./model_rl/iteration_{:03d}'.format(i+1))
        dql.generate_data_rl(100, 'data_rl/iteration_{:03d}'.format(i+1))
        vf = lambda x: dql.model_predict.predict(x)
        p1 = RLPlayer(vf, 0)
        p2 = RandomPlayer()
        print(compare_bots([p1, p2],50))
