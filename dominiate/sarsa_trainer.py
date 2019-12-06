from dominion import *
from rl_agent import *
import numpy as np
import tensorflow as tf
from cards import variable_cards
from game import Game, PlayerState, VICTORY_CARDS

class SarsaAgent():
    """
    Go back to our initial idea of training aggregated reward directly.
    No need for two net work. Just one network to fit for the aggregated reward for 
    each (s,a) pair
    """
    def __init__(self, epochs=10, length=129, gamma=0.99):
        self.epochs=epochs
        self.fit_iterations=10
        self.length = length
        # number of samples drawn every time
        self.mtrain = 1000
        self.gamma = gamma
        self.epsilon = 0.1
        self.create_model()
        self.data = []
        self.replaybuffer = 1000000
        self.win_reward = 100
        self.reward_points_per_turn = 1
        self.variable_cards = variable_cards

    def create_model(self, num_layers=3, dropout=0.2):
        layers = []
        for _ in range(num_layers - 1):
          layers.append(tf.keras.layers.Dense(self.length, activation='relu'))
          layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(tf.keras.layers.Dense(30, activation='relu'))
        layers.append(tf.keras.layers.Dropout(0.2))
        layers.append(tf.keras.layers.Dense(1, activation='linear'))
        model = tf.keras.models.Sequential(layers)
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        self.model = model
        return

    def create_model_5layers(self):
        return self.create_model(num_layers=5)

    def add_data(self, data):
        if self.data == []:
            self.data = data
        else:
            self.data = tuple([np.concatenate([d_this, data[i]]) for i,d_this in enumerate(self.data)])
        # truncate data down to replay buffer size
        if self.data[0].shape[0] > self.replaybuffer:
            print('truncate {:d} samples'.format(int(self.data[0].shape[0] - self.replaybuffer)))
            self.data = tuple([d_this[-int(self.replaybuffer):,:] for d_this in self.data])
        return

    def fit(self, data):
        """
        fit the buy network

        """
        # state, action, reward, next state
        s,a,r,n,ar = data
        sa = np.concatenate([s,a],axis=1)
        self.model.fit(sa, ar, epochs=self.epochs, verbose = 0)
        return

    def draw_sample(self):
        """
        draw random samples from the full dataset generated
        """
        m = self.data[1].shape[0]
        select = np.random.choice(m,self.mtrain,replace=False)
        return tuple([d[select,:] for d in self.data])

    def do_train(self):
        """
        Simply train for the aggregated rewards
        """
        for i in range(self.fit_iterations):
            self.fit(self.draw_sample())

    def save_model(self, fname='test'):
        self.model.save_weights(fname + '_ar.h5')
        return

    def load_model(self, fname='test'):
        self.model.load_weights(fname + '_ar.h5')
        return

    def generate_data_bot(self, bot, ngames = 50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model
        rl vs. specified bot
        """
        vf = lambda x: self.model.predict(x)
        p1 = RLPlayer(vf)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p1.include_action =  1
        bot.record_history = 0
        d_this = self.record_game(ngames, [p1,bot],fname)
        self.add_data(d_this)
        return d_this

    def generate_data(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. random bot
        """
        vf = lambda x: self.model.predict(x)
        p1 = RLPlayer(vf)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p1.include_action =  1
        p2 = RandomPlayer()
        p2.record_history = 0
        d_this = self.record_game(ngames, [p1,p2],fname)
        self.add_data(d_this)
        return d_this


    def generate_data_smithy(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. smithy bot
        """
        vf = lambda x: self.model.predict(x)
        p1 = RLPlayer(vf)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p1.include_action = 1
        p2 = smithyComboBotFactory()
        # try including smithy bot's data in the training.
        p2.record_history = 0
        d_this = self.record_game(ngames, [p1,p2],fname)
        self.add_data(d_this)
        return d_this


    def generate_data_rl(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. smithy bot
        """
        vf = lambda x: self.model.predict(x)
        p1 = RLPlayer(vf)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p1.include_action = 1
        p2 = RLPlayer(vf)
        p2.epsilon = self.epsilon
        p2.record_history = 1
        p2.include_action = 1
        d_this = self.record_game(ngames, [p1,p2],fname,  verbose=0)
        self.add_data(d_this)
        return d_this

    ###### Move dominion.py functions here for consistency
    # easier to modify for different fitting and reward functions

    def compare_bots(self, bots, num_games=50):
        start_time = time.time()
        wins = {bot: 0 for bot in bots}
        final_scores = {bot:[] for bot in bots}
        for i in range(num_games):
            random.shuffle(bots)
            game = Game.setup(bots, self.variable_cards)
            results = game.run()
            maxscore = 0
            for bot, score in results:
                final_scores[bot].append(score)
                if score > maxscore: maxscore = score
            for bot, score in results:
                if score == maxscore:
                    wins[bot] += 1
                    break
        for bot in final_scores.keys():
            final_scores[bot] = np.mean(final_scores[bot])
        print("Took %.3f seconds" % (time.time() - start_time))
        return wins, final_scores

    def run(self, players):
        game = Game.setup(players, self.variable_cards)
        # seems to have a bug that does not terminate game
        # set a limit of 5000 turns 
        k = 0
        while not game.over():
            if k >5000:
                print('terminate after 5000 turns!')
                break
            else:
                game = game.take_turn()
                k += 1
        scores = [(state.player, state.score()) for state in game.playerstates]
        winner, _ = max(scores, key=lambda item: item[1])
        loser, _ = min(scores, key=lambda item: item[1])
        winner.rewards[-1] += self.win_reward
        loser.rewards[-1] += -self.win_reward
        # add a reward that incentivize points per round
        for p, s in scores:
            p.rewards[-1] += 100*s/k*self.reward_points_per_turn
        return scores

    def scores_to_data(self, scores):
        """
        output history and reward in the form of numpy array
        Outputs all player if output_player is None. Otherwise, only output for the given player.
        turn the scores output from run() to X = (m, len(data vector)): the game state array
        and Y = (m, 1): the reward array
        where m is the number of states that were played through in the game
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        aggregated_rewards = []
        for player, fs in scores:
            if not player.record_history:
                continue
            states.append(player.states)
            actions.append(player.actions)
            rewards.append(player.rewards)
            next_states.append(player.choices) # just put something random here lol, legacy code
            ar = np.zeros_like(player.rewards,dtype=float)
            for i,r in enumerate(player.rewards[::-1]):
                ar[i] = r + self.gamma*ar[i-1]
            aggregated_rewards.append(ar[::-1])
        final_score = {bot: fs for bot, fs in scores}
        
        return np.concatenate(states), np.concatenate(actions), np.concatenate(rewards), np.concatenate(next_states),np.concatenate(aggregated_rewards), final_score

    def record_game(self, n, players, filename='', verbose=1):
        """
        play n games and save the results in filename
        save tuple (X,Y)
        X has size (m, len(data vector))
        Y has size (m, 1)
        m is the number of game states recorded
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        aggregated_rewards = []
        start_time = time.time()
        final_scores = {bot:[] for bot in players}
        for i in range(n):
            if i % 100 == 0:
              print("Playing game# %d" % i)
            # clear player history
            for p in players:
                p.reset_history()
            s, a, r, n, ar, fs = self.scores_to_data(self.run(players))
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n)
            aggregated_rewards.append(ar)
            for bot, fs_this in fs.items():
                final_scores[bot].append(fs_this)
        print("Took %.3f seconds" % (time.time() - start_time))
        # show the winrate of bots in the recorded games
        bot1, bot2 = final_scores.keys()
        bot1win = np.sum(np.array(final_scores[bot1]) > np.array(final_scores[bot2]))
        bot2win = len(final_scores[bot1]) - bot1win
        bot1avg = np.mean(final_scores[bot1])
        bot2avg = np.mean(final_scores[bot2])
        if verbose:
            print({bot1:bot1win, bot2:bot2win})
            print({bot1:bot1avg, bot2:bot2avg})
        # turn outputs into np array
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards).reshape([-1,1])
        next_states = np.concatenate(next_states)
        aggregated_rewards = np.concatenate(aggregated_rewards).reshape([-1,1])
        if not filename == '':
            with open(filename, 'wb') as f:
                pickle.dump((states, actions, rewards, next_states, aggregated_rewards), f)
        return states, actions, rewards, next_states, aggregated_rewards


    def load_game_data(self, filename):
        """
        load filename saved by record_game()
        """
        with open(filename, 'rb') as f:
            return  pickle.load(f)



class SarsaActBuyAgent(SarsaAgent):
    def __init__(self, epochs=10, length=129):
        self.epochs=epochs
        self.fit_iterations=10
        self.length = length
        # number of samples drawn every time
        self.mtrain = 1000
        self.gamma = 0.99
        self.epsilon = 0.1
        # don't create model in __init__
        # since we want to use self.record_data to know the data vector size first
        #self.create_model_5layers()
        #self.create_act_model()
        self.data = []
        self.replaybuffer = 1000000
        self.win_reward = 100
        self.reward_points_per_turn = 1
        self.VICTORY_CARDS = VICTORY_CARDS
        self.variable_cards = variable_cards

    def create_act_model(self):
        """
        Default with the deeper 5 layers model
        """
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(30, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        self.model_act = model
        return

    def run(self, players):
        game = SarsaActBuyAgent.setup(players, self.variable_cards, self.VICTORY_CARDS)
        # seems to have a bug that does not terminate game
        # set a limit of 5000 turns 
        k = 0
        while not game.over():
            if k >5000:
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
        if winner.rewards: winner.rewards[-1] += self.win_reward*win_reward_multiplier
        if loser.rewards: loser.rewards[-1] += -self.win_reward*win_reward_multiplier
        if winner.vp: winner.vp.append(winner.vp[-1] + self.win_reward*win_reward_multiplier)
        if loser.vp: loser.vp.append(loser.vp[-1] - self.win_reward*win_reward_multiplier)
        # add a reward that incentivize points per round
        for p, s in scores:
            p.rewards[-1] += 100*s/k*self.reward_points_per_turn
            # add a reward of points per turn to act_reward also
            if p.vp != []: p.vp[-1] += 100*s/k*self.reward_points_per_turn
        return scores

    def scores_to_data(self, scores):
        """
        output history and reward in the form of numpy array
        Outputs all player if output_player is None. Otherwise, only output for the given player.
        turn the scores output from run() to X = (m, len(data vector)): the game state array
        and Y = (m, 1): the reward array
        where m is the number of states that were played through in the game
        """
        states = []
        actions = []
        rewards = []
        aggregated_rewards = []
        states_act = []
        actions_act = []
        vp = []
        aggregated_rewards_act = []
        for player, fs in scores:
            if not player.record_history:
                continue
            # for buy phase
            states.append(player.states)
            actions.append(player.actions)
            rewards.append(player.rewards)
            ar = np.zeros_like(player.rewards,dtype=float)
            for i,r in enumerate(player.rewards[::-1]):
                ar[i] = r + self.gamma*ar[i-1]
            aggregated_rewards.append(ar[::-1])
            # for action phase
            if player.states_act: # if not empty
                states_act.append(player.states_act)
                actions_act.append(player.actions_act)
                vp.append(player.vp[:-1])
                ar = [player.vp[-1] - v for v in player.vp] # total vp minus current vp is Q(s,a)
                aggregated_rewards_act.append(ar[:-1])
        final_score = {bot: fs for bot, fs in scores}
        if states_act: 
            states_act = np.concatenate(states_act)
            actions_act = np.concatenate(actions_act)
            vp = np.concatenate(vp)
            aggregated_rewards_act = np.concatenate(aggregated_rewards_act)
        
        return np.concatenate(states), np.concatenate(actions), np.concatenate(rewards),np.concatenate(aggregated_rewards),\
               np.array(states_act), np.array(actions_act), np.array(vp), np.array(aggregated_rewards_act),\
                final_score

    def record_game(self, n, players, filename='', verbose=1):
        """
        play n games and save the results in filename
        save tuple (X,Y)
        X has size (m, len(data vector))
        Y has size (m, 1)
        m is the number of game states recorded
        """
        states = []
        actions = []
        rewards = []
        aggregated_rewards = []
        states_act = []
        actions_act = []
        rewards_act = []
        aggregated_rewards_act = []
        start_time = time.time()
        final_scores = {bot:[] for bot in players}
        for i in range(n):
            if i % 100 == 0:
              print("Playing game# %d" % i)
            # clear player history
            for p in players:
                p.reset_history()
            s, a, r, ar, s_act, a_act,r_act, ar_act, fs = self.scores_to_data(self.run(players))
            states.append(s)
            actions.append(a)
            rewards.append(r)
            aggregated_rewards.append(ar)
            states_act.append(s_act)
            actions_act.append(a_act)
            rewards_act.append(r_act)
            aggregated_rewards_act.append(ar_act)
            for bot, fs_this in fs.items():
                final_scores[bot].append(fs_this)
        print("Took %.3f seconds" % (time.time() - start_time))
        # show the winrate of bots in the recorded games
        bot1, bot2 = final_scores.keys()
        bot1win = np.sum(np.array(final_scores[bot1]) > np.array(final_scores[bot2]))
        bot2win = len(final_scores[bot1]) - bot1win
        bot1avg = np.mean(final_scores[bot1])
        bot2avg = np.mean(final_scores[bot2])
        if verbose:
            print({bot1:bot1win, bot2:bot2win})
            print({bot1:bot1avg, bot2:bot2avg})
        # turn outputs into np array
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards).reshape([-1,1])
        aggregated_rewards = np.concatenate(aggregated_rewards).reshape([-1,1])
        # remove empty arrays before concatenating
        select = [bool(s.size) for s in states_act]   # False for empty arrays
        states_act = np.concatenate(np.array(states_act)[select])
        actions_act = np.concatenate(np.array(actions_act)[select])
        # One dimensional arrays doesn't have to worry about this
        rewards_act = np.concatenate(rewards_act).reshape([-1,1])
        aggregated_rewards_act = np.concatenate(aggregated_rewards_act).reshape([-1,1])
        if filename != '':
            with open(filename, 'wb') as f:
                pickle.dump((states, actions, rewards,  aggregated_rewards, \
                            states_act, actions_act, rewards_act, aggregated_rewards_act), f)
        return states, actions, rewards,  aggregated_rewards, \
                            states_act, actions_act, rewards_act, aggregated_rewards_act

    def fit(self, data):
        """
        fit the buy network

        """
        # state, action, reward, next state
        s,a,r,ar,_,_,_,_ = data
        sa = np.concatenate([s,a],axis=1)
        self.model.fit(sa, ar, epochs=self.epochs, verbose = 0)
        return

    def fit_act(self, data):
        """
        fit the buy network

        """
        # state, action, reward, next state
        _,_,_,_,s,a,r,ar = data
        sa = np.concatenate([s,a],axis=1)
        self.model_act.fit(sa, ar, epochs=self.epochs, verbose = 0)
        return

    def draw_sample(self):
        """
        draw random samples from the full dataset generated
        """
        # for buy training 
        m = self.data[1].shape[0]
        select1 = np.random.choice(m,self.mtrain,replace=False)
        # for action training 
        m = self.data[4].shape[0]
        select2 = np.random.choice(m,self.mtrain,replace=False)
        return tuple([d[select1,:] for d in self.data[:4]] + [d[select2,:] for d in self.data[4:]])

    def do_train(self):
        """
        Simply train for the aggregated rewards
        For now train buy and act network the same number of times
        """
        for i in range(self.fit_iterations):
            self.fit(self.draw_sample())

    ###### below are functions for generating training data
    def generate_data(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. random bot
        """
        vbuy = lambda x: self.model.predict(x)
        vact = lambda x: self.model_act.predict(x)
        p1 = BuyActRLplayer(vbuy, vact)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p1.include_action =  1
        p2 = RandomPlayer()
        p2.record_history = 0
        d_this = self.record_game(ngames, [p1,p2],fname)
        self.add_data(d_this)
        return d_this


    def generate_data_smithy(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. smithy bot
        """
        vbuy = lambda x: self.model.predict(x)
        vact = lambda x: self.model_act.predict(x)
        p1 = BuyActRLplayer(vbuy, vact)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p1.include_action = 1
        p2 = SmithyBot()
        # try including smithy bot's data in the training.
        p2.record_history = 0
        d_this = self.record_game(ngames, [p1,p2],fname)
        self.add_data(d_this)
        return d_this


    def generate_data_rl(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. smithy bot
        """
        vbuy = lambda x: self.model.predict(x)
        vact = lambda x: self.model_act.predict(x)
        p1 = BuyActRLplayer(vbuy, vact)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p1.include_action = 1
        p2 = BuyActRLplayer(vbuy, vact)
        p2.epsilon = self.epsilon
        p2.record_history = 1
        p2.include_action = 1
        d_this = self.record_game(ngames, [p1,p2],fname,  verbose=1)
        self.add_data(d_this)
        return d_this

    def save_model(self, fname='test'):
        self.model.save_weights(fname + '_buy.h5')
        self.model_act.save_weights(fname + '_act.h5')
        return

    def load_model(self, fname='test'):
        self.model.load_weights(fname + '_buy.h5')
        self.model_act.load_weights(fname + '_act.h5')
        return

    def add_data(self, data):
        if self.data == []:
            self.data = data
        else:
            self.data = tuple([np.concatenate([d_this, data[i]]) for i,d_this in enumerate(self.data)])
        # truncate data down to replay buffer size
        if self.data[0].shape[0] > self.replaybuffer or self.data[4].shape[0] > self.replaybuffer:
            print('truncate {:d} samples'.format(int(self.data[0].shape[0] - self.replaybuffer)))
            self.data = tuple([d_this[-int(self.replaybuffer):,:] for d_this in self.data])
        return

    def compare_bots(self, bots, num_games=50):
        start_time = time.time()
        wins = {bot: 0 for bot in bots}
        final_scores = {bot:[] for bot in bots}
        for i in range(num_games):
            random.shuffle(bots)
            game = SarsaActBuyAgent.setup(bots, self.variable_cards, self.VICTORY_CARDS)
            results = game.run()
            maxscore = 0
            for bot, score in results:
                final_scores[bot].append(score)
                if score > maxscore: maxscore = score
            for bot, score in results:
                if score == maxscore:
                    wins[bot] += 1
                    break
        for bot in final_scores.keys():
            final_scores[bot] = np.mean(final_scores[bot])
        print("Took %.3f seconds" % (time.time() - start_time))
        return wins, final_scores

    @staticmethod
    def setup(players, var_cards=(), VICTORY_CARDS=VICTORY_CARDS, simulated=True):
        """Set up the game.
        Put this here because I want to try out different numbers of province. I'm hoping that in a longer game, 
        the AI can learn to play engine.
        """
        counts = {
            estate: VICTORY_CARDS[len(players)],
            duchy: VICTORY_CARDS[len(players)],
            province: VICTORY_CARDS[len(players)],
            curse: 10*(len(players)-1),
            copper: 60 - 7*len(players),
            silver: 40,
            gold: 30
        }
        for card in var_cards:
            counts[card] = 10

        playerstates = [PlayerState.initial_state(p) for p in players]
        random.shuffle(playerstates)
        return Game(playerstates, counts, turn=0, simulated=simulated)

######## Below are scripts to evaluate model evolution


