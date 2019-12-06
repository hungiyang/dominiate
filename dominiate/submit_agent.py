# write a function to submit training jobs instead of using run_sarsa.py
from game import Game
from players import HumanPlayer
from basic_ai import SmithyBot
from cards import variable_cards
from basic_ai import SmithyBot
from rl_agent import RLPlayer, RandomPlayer, BuyActRLplayer
import numpy as np
import pickle as pk
import tensorflow as tf
from dqltrainer import DQLagent, DQLSarsaAgent
from dqlvaluetrainer import DQLValueAgent, ARagent
from sarsa_trainer import SarsaAgent
from sarsa_on_policy_trainer import SarsaBootstrapAgent

def run_agent(fn = 'sarsa_v0',version = 'SarsaBootstrapAgent', gamma_comp = 0.02, epsilon = 0.05, dropout = 0.2):
  if version == 'SarsaBootstrapAgent':
    dql = SarsaBootstrapAgent()
  elif version == 'DQLSarsaAgent':
    dql = DQLSarsaAgent()
  p1 = SmithyBot()
  p1.record_history = 1
  p2 = SmithyBot()
  p2.record_history = 0
  dbuy, _ = dql.record_game(1, [p1, p2])
  sa = np.array([np.concatenate([s, a]) for s, a, r, _, _, _ in dbuy])
  r = np.array([r for s, a, r, _, _, _ in dbuy])

  dql.epsilon = 0.05
  dql.mtrain = 5000
  # one iteration creates roughly 1e4 samples
  # therefore this remembers the data of pass 40 iterations.
  dql.replaybuffer = int(4e5)
  dql.target_iterations = 1
  dql.predict_iterations = 10
  dql.epochs = 10
  dql.gamma = 1- gamma_comp
  # incentivize short games
  dql.reward_points_per_turn = 0.0
  # I think having win reward makes it too noisy
  dql.win_reward = 0
  # print the settings
  print('mtrain {:d}, replaybuffer {:d}, predict iter {:d}, epochs {:d}, gamma {:.02f}'.\
      format(dql.mtrain, dql.replaybuffer, dql.predict_iterations, dql.epochs, dql.gamma))
  # a very dumb way to initiate the network weights
  dql.length = sa.shape[0]
  dql.create_model(sa, r, dropout = dropout)
  # start training iterations.
  for i in range(1000):
    print('data generation iteration {:d}'.format(i))
    dql.epsilon = 0.05 / ((i + 1) / 20)
    print('dql epsilon: {:.04f}'.format(dql.epsilon))
    dql.generate_data_smithy(100)
    dql.generate_data_rl(100)
    #dql.generate_data(100)
    print('data sample size = {:d}'.format(dql.data.shape[0]))
    dql.do_target_iteration()
    dql.save_model('./model/{:s}_iteration_{:03d}'.format(fn,i+1))
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model_predict.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))
    # print output of Q(s,a) estimates for the first SmithyBot game
    print('Q(s,a) estimates of a SmithyBot game')
    print(dql.model_predict.predict(sa).T)


if __name__ == '__main__':
  run_agent()



