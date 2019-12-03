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
from dqltrainer import DQLagent, DQLSalsaAgent
from dqlvaluetrainer import DQLValueAgent, ARagent
from sarsa_trainer import SarsaAgent

# Train against rlbot itself/smithy bot as opponent, deeper network
# New training scheme with bootstrap. DQLSalsaAgent in dqltrainer.py
if 1:
  dql = DQLSalsaAgent()
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  dbuy, _ = dql.record_game(1, [p1,p2])
  sa = np.array([np.concatenate([s,a]) for s,a,r,_,_,_ in dbuy])
  r = np.array([r for s,a,r,_,_,_ in dbuy])

  # a very dumb way to initiate the network weights
  dql.create_model(sa, r)
  dql.epsilon = 0.05
  dql.mtrain = 5000
  # one iteration creates roughly 1e4 samples
  # therefore this remembers the data of pass 40 iterations.
  dql.replaybuffer = int(4e5)
  dql.target_iterations=1
  dql.predict_iterations=20
  dql.epochs = 50
  # incentivize short games
  dql.reward_points_per_turn = 0.0
  # I think having win reward makes it too noisy
  dql.win_reward = 0

# use dql vs. random player's game log to train
for i in range(1000):
  dql.epsilon = 0.05/((i+1)/20) 
  print('dql epsilon: {:.04f}'.format(dql.epsilon))
  print('data generation iteration {:d}'.format(i))
  dql.generate_data_smithy(100)
  dql.generate_data_rl(100)
  dql.generate_data(100)
  print('data sample size = {:d}'.format(dql.data[0].shape[0]))
  dql.do_target_iteration()
  dql.save_model('./model/DQLSarsa_buy_only_v0_iteration_{:03d}'.format(i+1))
  # evaluate against random bot and smithy bot
  p1 = RLPlayer(lambda x: dql.model_predict.predict(x))
  p1.epsilon=0.0
  p_smith = SmithyBot()
  print(compare_bots([p1, RandomPlayer()],10))
  print(compare_bots([p1, p_smith],10))


###### train against rlbot itself/smithy bot as opponent, deeper network
# New training scheme with bootstrap. DQLSalsaAgent in dqltrainer.py
if 0:
    dql = DQLSalsaAgent()
    p1 = RandomPlayer()
    p1.record_history = 1
    p2 = RandomPlayer()
    p2.record_history = 1
    dbuy, _ = dql.record_game(1, [p1,p2])
    sa = np.array([np.concatenate([s,a]) for s,a,r,_,_,_ in dbuy])
    r = np.array([r for s,a,r,_,_,_ in dbuy])

# a very dumb way to initiate the network weights
    dql.create_model(sa, r)
    dql.epsilon = 0.05
    dql.mtrain = 1000
# one iteration creates roughly 1e4 samples
# therefore this remembers the data of pass 40 iterations.
    dql.replaybuffer = 4e5
    dql.target_iterations=1
    dql.predict_iterations=10
    dql.epochs = 10
    # incentivize short games
    dql.reward_points_per_turn = 0.0
    # I think having win reward makes it too noisy
    dql.win_reward = 0


# use dql vs. random player's game log to train
    for i in range(1000):
        dql.epsilon = 0.05/((i+1)/20) 
        print('dql epsilon: {:.04f}'.format(dql.epsilon))
        print('data generation iteration {:d}'.format(i))
        dql.generate_data_smithy(100)
        dql.generate_data_rl(100)
        dql.generate_data(100)
        print('data sample size = {:d}'.format(dql.data[0].shape[0]))
        dql.do_target_iteration()
        dql.save_model('./model/DQLSarsa_buy_only_v0_iteration_{:03d}'.format(i+1))
        # evaluate against random bot and smithy bot
        p1 = RLPlayer(lambda x: dql.model_predict.predict(x))
        p1.epsilon=0.0
        p_smith = SmithyBot()
        print(compare_bots([p1, RandomPlayer()],10))
        print(compare_bots([p1, p_smith],10))


####### 
###### train against rlbot itself/smithy bot as opponent, deeper network
# RL bot now also decides what actions to play on its own
# use no win reward and no reward per turn
# The previous act network(before v2) have bug and does not work.
if 0:
  dql = SarsaActBuyAgent()
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = dql.record_game(500, [p1, p2])

  # set the network size to the input vector length
  dql.length = (data[0].shape[1] + data[1].shape[1])
  # run with a deeper network
  dql.create_model_5layers()
  dql.create_act_model()
  dql.epsilon = 0.05
  dql.mtrain = 5000
  # one iteration creates roughly 1e4 samples
  # therefore this remembers the data of pass 40 iterations.
  dql.replaybuffer = 4e5
  dql.fit_iterations = 10
  dql.epochs = 10
  # incentivize short games
  dql.reward_points_per_turn = 0.0
  # I think having win reward makes it too noisy
  dql.win_reward = 0

  # first training step
  dql.add_data(data)
  dql.do_train()

  # use dql vs. random player's game log to train
  for i in range(1000):
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 10)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/BuyActRL_v3_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(150)
    dql.generate_data_rl(150)
    # few games with random bot
    dql.generate_data(50)
    # evaluate against random bot and smithy bot
    p1 = BuyActRLplayer(lambda x: dql.model.predict(x),
                        lambda x: dql.model_act.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

#######
###### train against rlbot itself/smithy bot as opponent, deeper network
# RL bot now also decides what actions to play on its own
# use no win reward and no reward per turn
# Result is stronger than SmithyBot!!!
if 0:
  # What happens when we have 20 province to play with?
  VICTORY_CARDS = {2: 20}
  # I think it is fine for CARD_VECTOR_ORDER to say the same
  # There will just be some zeros in the vector
  variable_cards = [
      village, cellar, smithy, festival, market, laboratory, chapel, warehouse,
      council_room, militia, moat
  ]

  dql = SarsaActBuyAgent()
  dql.variable_cards = variable_cards
  dql.VICTORY_CARDS = VICTORY_CARDS

  # initial training data
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = dql.record_game(500, [p1, p2])

  # set the network size to the input vector length
  dql.length = (data[0].shape[1] + data[1].shape[1])
  # run with a deeper network
  dql.create_model_5layers()
  dql.create_act_model()
  dql.epsilon = 0.05
  dql.mtrain = 5000
  # one iteration creates roughly 1e4 samples
  # therefore this remembers the data of pass 40 iterations.
  dql.replaybuffer = 4e5
  dql.fit_iterations = 10
  dql.epochs = 10
  # incentivize short games
  dql.reward_points_per_turn = 0.0
  # I think having win reward makes it too noisy
  dql.win_reward = 0

  # first training step
  dql.add_data(data)
  dql.do_train()

  # use dql vs. random player's game log to train
  for i in range(1000):
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 20)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model(
        './model/BuyActRL_20_province_no_witch_v0_iteration_{:03d}'.format(i +
                                                                           1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(150)
    # dql.generate_data(100)
    dql.generate_data_rl(150)
    # evaluate against random bot and smithy bot
    p1 = BuyActRLplayer(lambda x: dql.model.predict(x),
                        lambda x: dql.model_act.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

#######
###### train against rlbot itself/smithy bot as opponent, deeper network
# RL bot now also decides what actions to play on its own
# use no win reward and no reward per turn
# Result is stronger than SmithyBot!!!
# However, it ends up with so few action card that the action network is pretty much useless
if 0:
  dql = SarsaActBuyAgent()
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = dql.record_game(500, [p1, p2])

  # set the network size to the input vector length
  dql.length = (data[0].shape[1] + data[1].shape[1])
  # run with a deeper network
  dql.create_model_5layers()
  dql.create_act_model()
  dql.epsilon = 0.05
  dql.mtrain = 5000
  # one iteration creates roughly 1e4 samples
  # therefore this remembers the data of pass 40 iterations.
  dql.replaybuffer = 4e5
  dql.fit_iterations = 10
  dql.epochs = 10
  # incentivize short games
  dql.reward_points_per_turn = 0.0
  # I think having win reward makes it too noisy
  dql.win_reward = 0

  # first training step
  dql.add_data(data)
  dql.do_train()

  # use dql vs. random player's game log to train
  for i in range(1000):
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 20)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/BuyActRL_v2_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(150)
    # dql.generate_data(100)
    dql.generate_data_rl(150)
    # evaluate against random bot and smithy bot
    p1 = BuyActRLplayer(lambda x: dql.model.predict(x),
                        lambda x: dql.model_act.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

#######
###### train against rlbot itself/smithy bot as opponent, deeper network
# RL bot now also decides what actions to play on its own
# use win reward and no reward per turn
# converges super slowly.. why?
if 0:
  dql = SarsaActBuyAgent()
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = dql.record_game(500, [p1, p2])

  # set the network size to the input vector length
  dql.length = (data[0].shape[1] + data[1].shape[1])
  # run with a deeper network
  dql.create_model_5layers()
  dql.create_act_model()
  dql.epsilon = 0.05
  dql.mtrain = 5000
  # one iteration creates roughly 1e4 samples
  # therefore this remembers the data of pass 40 iterations.
  dql.replaybuffer = 4e5
  dql.fit_iterations = 10
  dql.epochs = 10
  # incentivize short games
  dql.reward_points_per_turn = 0.0
  # I think having win reward makes it too noisy
  dql.win_reward = 20

  # first training step
  dql.add_data(data)
  dql.do_train()

  # use dql vs. random player's game log to train
  for i in range(1000):
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 20)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/BuyActRL_v1_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(150)
    # dql.generate_data(100)
    dql.generate_data_rl(150)
    # evaluate against random bot and smithy bot
    p1 = BuyActRLplayer(lambda x: dql.model.predict(x),
                        lambda x: dql.model_act.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

#######
###### train against rlbot itself/smithy bot as opponent, deeper network, incentivize short games
# RL bot now also decides what actions to play on its own
if 0:
  dql = SarsaActBuyAgent()
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = dql.record_game(500, [p1, p2])

  # set the network size to the input vector length
  dql.length = (data[0].shape[1] + data[1].shape[1])
  # run with a deeper network
  dql.create_model_5layers()
  dql.create_act_model()
  dql.epsilon = 0.05
  dql.mtrain = 5000
  # one iteration creates roughly 1e4 samples
  # therefore this remembers the data of pass 40 iterations.
  dql.replaybuffer = 4e5
  dql.fit_iterations = 10
  dql.epochs = 10
  # incentivize short games
  dql.reward_points_per_turn = 0.5
  # I think having win reward makes it too noisy
  dql.win_reward = 0

  # first training step
  dql.add_data(data)
  dql.do_train()

  # use dql vs. random player's game log to train
  for i in range(1000):
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 20)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/BuyActRL_v0_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(150)
    # dql.generate_data(100)
    dql.generate_data_rl(150)
    # evaluate against random bot and smithy bot
    p1 = BuyActRLplayer(lambda x: dql.model.predict(x),
                        lambda x: dql.model_act.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

####### add witch into the available cards
###### train against rlbot itself/random bot/smithy bot as opponent, deeper network, incentivize short games
# change the act priority calculation
# add non buy decisions into training data
if 0:
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = record_game(1000, [p1, p2])
  #data = load_game_data('data/1000with_action')

  # set the network size to the input vector length
  dql = SarsaAgent(length=(data[0].shape[1] + data[1].shape[1]))
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
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 10)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model(
        './model/combination_with_witch_v1_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(100)
    dql.generate_data(100)
    dql.generate_data_rl(100)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

####### add witch into the available cards
###### train against rlbot itself/random bot/smithy bot as opponent, deeper network, incentivize short games
# TODO: change the order of actions card to play so that witch is prioritized over smithy etc.
if 0:
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = record_game(1000, [p1, p2])
  #data = load_game_data('data/1000with_action')

  # set the network size to the input vector length
  dql = SarsaAgent(length=(data[0].shape[1] + data[1].shape[1]))
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
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 20)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model(
        './model/combination_with_witch_v0_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(100)
    dql.generate_data(100)
    dql.generate_data_rl(100)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

if 0:
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
  dql = SarsaAgent(length=(data[0].shape[1] + data[1].shape[1]))
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
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 10)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/rand_v1_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data(500)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    p_smith = smithyComboBotFactory()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

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
  dql = SarsaAgent(length=(data[0].shape[1] + data[1].shape[1]))
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
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 10)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/combination_v1_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(100)
    dql.generate_data(100)
    dql.generate_data_rl(100)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    p_smith = smithyComboBotFactory()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

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
  dql = SarsaAgent(length=(data[0].shape[1] + data[1].shape[1]))
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
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 50)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/combination_v0_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(100)
    dql.generate_data(100)
    dql.generate_data_rl(100)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    p_smith = smithyComboBotFactory()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

if 0:
  ###### train against smithy bot, deeper network, incentivize short games
  # start with random player's game log
  # This got stuck somehow and didn't proceed
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = smithyComboBotFactory()
  p2.record_history = 0
  data = record_game(1000, [p1, p2])

  # set the network size to the input vector length
  dql = SarsaAgent(length=(data[0].shape[1] + data[1].shape[1]))
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
    if i % 100 == 0 and i != 0:
      dql.epsilon = 0.05 / (i / 50)
      print('dql epsilon: {:.04f}'.format(dql.epsilon))
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/smithy_v0_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_smithy(500)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    p_smith = smithyComboBotFactory()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

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
  dql = SarsaAgent(length=(data[0].shape[1] + data[1].shape[1]))
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
    if i == 100:
      dql.epsilon = 0.04
      print('dql epsilon: 0.04')
    elif i == 200:
      dql.epsilon = 0.03
      print('dql epsilon: 0.04')
    elif i == 300:
      dql.epsilon = 0.02
      print('dql epsilon: 0.02')
    elif i == 400:
      dql.epsilon = 0.01
      print('dql epsilon: 0.01')
    print('data sample size = {:d}'.format(dql.data[0].shape[0]))
    dql.do_train()
    dql.save_model('./model/rand_v0_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data(500)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    p_smith = SmithyBot()
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, p_smith], 10))

if 0:
  # Continue with the RL vs. RL bot start with random player's game log.
  p1 = RandomPlayer()
  p1.record_history = 1
  p2 = RandomPlayer()
  p2.record_history = 1
  data = record_game(1, [p1, p2])

  # set the network size to the input vector length
  dql = ARagent(length=(data[0].shape[1] + data[1].shape[1]))
  dql.fit(data)  # to establish the network
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
    dql.save_model('./model/rl_v1_iteration_{:03d}'.format(i + 1))
    print('data generation iteration {:d}'.format(i))
    dql.generate_data_rl(200)
    # evaluate against random bot and smithy bot
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.epsilon = 0.0
    print(compare_bots([p1, RandomPlayer()], 10))
    print(compare_bots([p1, SmithyBot()], 10))
