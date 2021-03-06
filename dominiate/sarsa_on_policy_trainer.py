from rl_agent import RLPlayer, RandomPlayer, BuyActRLplayer
import numpy as np
import tensorflow as tf
from cards import copper, silver, gold, curse, estate, duchy, province, CARD_VECTOR_ORDER,variable_cards
from game import Game, PlayerState, VICTORY_CARDS
import time
import random

class SarsaBootstrapAgent():
  """
    Bootstrap DQL method in DQLSarsaAgent does not work that well
    Try using on plicy 
    """

  def __init__(self, epochs=10, gamma=0.99):
    self.epochs = epochs
    self.target_iterations = 5
    self.predict_iterations = 200
    # number of samples drawn every time
    self.mtrain = 2000
    self.gamma = gamma
    self.epsilon = 0.1
    #self.create_model()
    self.data = []
    self.replaybuffer = 4e5
    self.win_reward = 0
    self.reward_points_per_turn = 0
    self.VICTORY_CARDS = VICTORY_CARDS
    self.variable_cards = variable_cards

  def create_model(self, sa, target, num_layers=3, dropout=0.2):
    def _make_model(num_layers, dropout):
        layers = []
        for _ in range(num_layers - 1):
          layers.append(tf.keras.layers.Dense(128, activation='relu'))
          layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(30, activation='relu'))
        layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(1, activation='linear'))
        model = tf.keras.models.Sequential(layers)
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model
    model = _make_model(num_layers, dropout)
    # initiate network
    model.fit(sa, target, epochs=1, verbose=1)
    self.model_predict = model
    # target network
    model = _make_model(num_layers, dropout)
    # initiate network
    model.fit(sa, target, epochs=1, verbose=1)
    self.model_target = model
    return

  def run(self, players):
    game = self.setup(players, self.variable_cards, self.VICTORY_CARDS)
    # seems to have a bug that does not terminate game
    # set a limit of 5000 turns
    k = 0
    while not game.over():
      if k > 5000:
        print('terminate after 5000 turns!')
        break
      else:
        game = game.take_turn()
        k += 1
    scores = [(state.player, state.score()) for state in game.playerstates]
    # This code is buggy... It picks the same guy when there's a tie
    #winner, _ = max(scores, key=lambda item: item[1])
    #loser, _ = max(scores, key=lambda item: item[1])
    # write stupid code instead. No win reward when there's a tie
    if scores[0][1] > scores[1][1]:
      winner = scores[0][0]
      loser = scores[1][0]
      win_reward_multiplier = 1
    elif scores[0][1] == scores[1][1]:
      win_reward_multiplier = 0
      winner = scores[0][0]
      loser = scores[1][0]
    else:
      winner = scores[1][0]
      loser = scores[0][0]
      win_reward_multiplier = 1
    # check if list is empty before adding reward
    if winner.rewards:
      winner.rewards[-1] += self.win_reward * win_reward_multiplier
    if loser.rewards:
      loser.rewards[-1] += -self.win_reward * win_reward_multiplier
    # append to vp the win_reward
    if winner.vp:
      winner.vp.append(self.win_reward * win_reward_multiplier)
    if loser.vp:
      loser.vp.append(self.win_reward * win_reward_multiplier)
    # add a reward that incentivize points per round
    for p, s in scores:
      p.rewards[-1] += 100 * s / k * self.reward_points_per_turn
      # add a reward of points per turn to act_reward also
      if p.vp != []:
        p.vp[
            -1] += s  # add the final score to additional element of vp. this is used to calculate the reward for the final s,a
        p.vp[-1] += 100 * s / k * self.reward_points_per_turn
    return scores

  def scores_to_data(self, scores):
    """
        Each data point consistes of a tuple (s,a,r,s',a',a'_opt)
        s: state
        a: action
        r: reward
        s': next state
        a'_opt: next state
        last_marker: 1 if it's the last state, 0 if not
        What do we do for the last step?
        set the training target to zero I guess

        """
    for player, fs in scores:
      if not player.record_history:
        continue
      # for buy phase
      d_buy = []
      for i in range(len(player.states) - 1):
        s = np.array(player.states[i])
        a = np.array(player.actions[i])
        r = np.array(player.rewards[i])
        sn = np.array(player.states[i + 1])
        an = np.array(player.actions[i + 1]) # next action
        d_buy.append((s, a, r, sn, an,
                      0))  # 0 means that it is not the end of the sequence
      # deal with the last state
      s = np.array(player.states[-1])
      a = player.actions[-1]
      r = np.array(player.rewards[-1])
      d_buy.append((s, a, r, None, None, 1))
      # for action phase
      d_act = []
      if player.states_act:  # if not empty
        for i in range(len(player.states_act) - 1):
          s_act = np.array(player.states_act[i])
          a_act = player.actions_act[i]
          r_act = np.array(player.vp[i + 1] - player.vp[i])
          sn_act = np.array(player.states_act[i + 1])
          an_act = np.array(player.actions_act[i + 1])
          d_act.append((s_act, a_act, r_act, sn_act, an_act,
                        0))  # 0 means not the end of sequence
        # deal with the last state, in run() function, we already appended the final score and terminal reward to player.vp
        # so player.vp is one longer than the other lists
        s_act = np.array(player.states_act[-1])
        a_act = player.actions_act[-1]
        r_act = np.array(player.vp[-1] - player.vp[-2])
        d_act.append((s_act, a_act, r_act, None, None, 1))
    final_score = {bot: fs for bot, fs in scores}

    return d_buy, d_act, final_score

  def record_game(self, n, players, filename='', verbose=1):
    """
        run n games and record to data
        use the output of scores_to_data
        """
    start_time = time.time()
    final_scores = {bot: [] for bot in players}
    d_buy = []
    d_act = []
    for i in range(n):
      if i % 100 == 0:
        print('Playing game# %d' % i)
      # clear player history
      for p in players:
        p.reset_history()
      db_this, da_this, fs = self.scores_to_data(self.run(players))
      d_buy += db_this
      d_act += da_this
      for bot, fs_this in fs.items():
        final_scores[bot].append(fs_this)
    print('Took %.3f seconds' % (time.time() - start_time))
    # show the winrate of bots in the recorded games
    bot1, bot2 = final_scores.keys()
    bot1win = np.sum(
        np.array(final_scores[bot1]) > np.array(final_scores[bot2]))
    bot2win = len(final_scores[bot1]) - bot1win
    bot1avg = np.mean(final_scores[bot1])
    bot2avg = np.mean(final_scores[bot2])
    if verbose:
      print({bot1: bot1win, bot2: bot2win})
      print({bot1: bot1avg, bot2: bot2avg})
    if filename != '':
      with open(filename, 'wb') as f:
        pickle.dump((d_buy, d_act), f)
    return (d_buy, d_act)

  def compute_target_old(self, data):
    """
        compute_target use the target network to predict the Q value
        n is the next state
        with a Q(s,a) model
        compute r + gamma*max_a' Q(s',a')
        It outputs the target that the deep neural network wants to fit for.
        a' are the possible actions
        """
    sa = []
    target = []
    for (s, a, r, ns, na, isend) in data:
      sa.append(np.concatenate([s, a]))
      if not isend:
        qn = self.model_target.predict(np.concatenate([ns, na]).reshape(1,-1))
        target.append(r + self.gamma * qn)
      else:
        target.append(r)
    sa = np.array(sa)
    target = np.array(target)
    return sa, target

  def compute_target(self, data):
    """
        compute_target use the target network to predict the Q value
        n is the next state
        with a Q(s,a) model
        compute r + gamma*max_a' Q(s',a')
        It outputs the target that the deep neural network wants to fit for.
        a' are the possible actions
        Try to parallelize the model.predict() part
        this is 10 times faster that the old code
        """
    safull = []
    ind = []
    target = []
    sa = []
    for i, (s, a, r, ns, na, isend) in enumerate(data):
      sa.append(np.concatenate([s, a]))
      target.append(float(r))
      if not isend:
        safull.append(np.concatenate([ns, na]).reshape(1, -1))
        ind.append(i)  # record which data point each sa belongs to
    safull = np.concatenate(safull)
    ind = np.asarray(ind, dtype=int)
    Qp = self.model_target.predict(safull)
    for i in np.unique(ind):  # loop over data point i that are not terminal s,a
      # pretty inefficient, but shouldn't be the bottle neck
      target[i] += self.gamma * Qp[np.where(ind == i)[0]]
    target = np.array(target)
    sa = np.array(sa)
    return sa, target


  def draw_sample(self):
    """
        draw random samples from the full dataset generated
        """
    # for buy training
    m = self.data[1].shape[0]
    select1 = np.random.choice(m, self.mtrain, replace=False)
    # for action training
    m = self.data[4].shape[0]
    select2 = np.random.choice(m, self.mtrain, replace=False)
    return tuple([d[select1, :] for d in self.data[:4]] +
                 [d[select2, :] for d in self.data[4:]])

  ###### below are functions for generating training data
  def generate_data(self, ngames=50, fname=''):
    """
        generate a new batch of data with the latest prediction model
        self.model_predict
        rl vs. random bot
        """
    vbuy = lambda x: self.model_predict.predict(x)
    # vact = lambda x: self.model_act.predict(x)
    # p1 = BuyActRLplayer(vbuy, vact)
    p1 = RLPlayer(vbuy)
    p1.epsilon = self.epsilon
    p1.record_history = 1
    p1.include_action = 1
    p2 = RandomPlayer()
    p2.record_history = 0
    d_this, _ = self.record_game(ngames, [p1, p2], fname)
    self.add_data(d_this)
    return d_this

  def generate_data_smithy(self, ngames=50, fname=''):
    """
        generate a new batch of data with the latest prediction model
        self.model_predict
        rl vs. smithy bot
        """
    vbuy = lambda x: self.model_predict.predict(x)
    # vact = lambda x: self.model_act.predict(x)
    # p1 = BuyActRLplayer(vbuy, vact)
    p1 = RLPlayer(vbuy)
    p1.epsilon = self.epsilon
    p1.record_history = 1
    p1.include_action = 1
    p2 = SmithyBot()
    # try including smithy bot's data in the training.
    p2.record_history = 0
    d_this, _ = self.record_game(ngames, [p1, p2], fname)
    self.add_data(d_this)
    return d_this

  def generate_data_rl(self, ngames=50, fname=''):
    """
        generate a new batch of data with the latest prediction model
        self.model_predict
        rl vs. smithy bot
        """
    vbuy = lambda x: self.model_predict.predict(x)
    #vact = lambda x: self.model_act.predict(x)
    # p1 = BuyActRLplayer(vbuy, vact)
    p1 = RLPlayer(vbuy)
    p1.epsilon = self.epsilon
    p1.record_history = 1
    p1.include_action = 1
    p2 = RLPlayer(vbuy)
    p2.epsilon = self.epsilon
    p2.record_history = 1
    p2.include_action = 1
    d_this, _ = self.record_game(ngames, [p1, p2], fname, verbose=1)
    self.add_data(d_this)
    return d_this

  def save_model(self, fname='test'):
    self.model_predict.save_weights(fname + '_predict.h5')
    self.model_target.save_weights(fname + '_target.h5')
    return

  def load_model(self, fname='test'):
    self.model_predict.load_weights(fname + '_predict.h5')
    self.model_target.load_weights(fname + '_target.h5')
    return

  def compare_bots(self, bots, num_games=50):
    start_time = time.time()
    wins = {bot: 0 for bot in bots}
    final_scores = {bot: [] for bot in bots}
    for i in range(num_games):
      random.shuffle(bots)
      game = self.setup(bots, self.variable_cards, self.VICTORY_CARDS)
      results = game.run()
      maxscore = 0
      for bot, score in results:
        final_scores[bot].append(score)
        if score > maxscore:
          maxscore = score
      for bot, score in results:
        if score == maxscore:
          wins[bot] += 1
          break
    for bot in final_scores.keys():
      final_scores[bot] = np.mean(final_scores[bot])
    print('Took %.3f seconds' % (time.time() - start_time))
    return wins, final_scores

  def fit_target(self, data):
    """Fit_target_network.

    Computes the target network prediction and fit for it with prediction
    network.
    """
    # state, action, reward, next state
    sa, target = self.compute_target(data)
    self.model_predict.fit(sa, target, epochs=self.epochs, verbose=1)
    return

  def do_target_iteration(self):
    for j in range(self.target_iterations):
      #print('start target model iteration {:d}'.format(j))
      # set the weights of the target model to predict model
      self.model_target.set_weights(self.model_predict.get_weights())
      for i in range(self.predict_iterations):
        print('prediction model iteration {:d}'.format(i))
        self.fit_target(self.draw_sample())

  def draw_sample(self):
    """Draws random samples from the full dataset generated."""
    m = len(self.data)
    select = np.random.choice(m, self.mtrain, replace=False)
    return self.data[select]

  def add_data(self, data):
    if self.data == []:
      self.data = np.array(data)
    else:
      self.data = np.concatenate([self.data, np.array(data)])
    # truncate data down to replay buffer size
    if self.data.shape[0] > self.replaybuffer:
      print('truncate {:d} samples'.format(self.data.shape[0] -
                                           self.replaybuffer))
      self.data = self.data[-self.replaybuffer:]
    return

  def add_data_act(self, data):
    if self.data_act == []:
      self.data_act = np.array(data)
    else:
      self.data_act = np.concatenate([self.data_act, np.array(data)])
    # truncate data_act down to replay buffer size
    if self.data_act.shape[0] > self.replaybuffer:
      print('truncate {:d} samples'.format(self.data_act.shape[0] -
                                           self.replaybuffer))
      self.data_act = self.data_act[-self.replaybuffer:]
    return

  def setup(self, players, var_cards=(), VICTORY_CARDS=VICTORY_CARDS, simulated=True):
    """Set up the game.

    Put this here because I want to try out different numbers of province. I'm
    hoping that in a longer game,
    the AI can learn to play engine.
    """
    counts = {
        estate: VICTORY_CARDS[len(players)],
        duchy: VICTORY_CARDS[len(players)],
        province: VICTORY_CARDS[len(players)],
        curse: 10 * (len(players) - 1),
        copper: 60 - 7 * len(players),
        silver: 40,
        gold: 30
    }
    counts.update({card: 10 for card in var_cards})
    playerstates = [PlayerState.initial_state(p) for p in players]
    random.shuffle(playerstates)
    return Game(playerstates, counts, turn=0, simulated=simulated)
