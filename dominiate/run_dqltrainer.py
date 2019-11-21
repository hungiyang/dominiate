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
data = record_game(500, [p1,p2], 'data/iteration_0')
data0 = data

dql = DQLagent()
dql.target_iterations=50
dql.predic_iterations=50
dql.do_target_iteration(data)

# use dql vs. random player's game log to train
dql.target_iterations=30
dql.predic_iterations=30
for i in range(100):
    data = dql.generate_data(100, 'data/iteration_{:03d}'.format(i+1))
    dql.do_target_iteration(data)
    # dql.save_weights('model/iteration_{:03d}'.format(i+1))
    vf = lambda x: dql.model_predict.predict(x)
    p1 = RLPlayer(vf)
    p2 = RandomPlayer()
    print(compare_bots([p1, p2],10))
