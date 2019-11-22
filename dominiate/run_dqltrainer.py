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
# data = record_game(1000, [p1,p2], 'data/iteration_0')
# load 5000 games of presaved random
data = load_game_data('data/5000random')

# set the network size to the input vector length
dql = DQLagent(length=(data[0].shape[1]+data[1].shape[1]))
dql.add_data(data)
dql.mtrain = 1000
dql.target_iterations=5
dql.predict_iterations=200

# use dql vs. random player's game log to train
for i in range(1000):
    print('data generation iteration {:d}'.format(i))
    dql.do_target_iteration()
    dql.save_model('./model/iteration_{:03d}'.format(i+1))
    dql.generate_data(100, 'data/iteration_{:03d}'.format(i+1))
