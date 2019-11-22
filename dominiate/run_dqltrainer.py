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

# start with random player's game log
p1 = RandomPlayer()
p1.record_history = 1
p2 = RandomPlayer()
p2.record_Plahistory = 1
data = record_game(1000, [p1,p2], 'data/iteration_0')

# set the network size to the input vector length
dql = DQLagent(length=(data[0].shape[1]+data[1].shape[1]))
dql.mtrain = 1000
dql.target_iterations=50
dql.predic_iterations=200
dql.do_target_iteration(data)

# use dql vs. random player's game log to train
dql.target_iterations=50
dql.predic_iterations=200
for i in range(1000):
    print('data generation iteration {:d}'.format(i))
    data = dql.generate_data(500, 'data2/iteration_{:03d}'.format(i+1))
    dql.do_target_iteration(data)
    # dql.save_weights('model/iteration_{:03d}'.format(i+1))
    vf = lambda x: dql.model_predict.predict(x)
    # evaluate the model with the epsilon greedy part
    p1 = RLPlayer(vf,0)
    p2 = RandomPlayer()
    print(compare_bots([p1, p2],20))
