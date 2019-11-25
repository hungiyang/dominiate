from dominion import *
from rl_agent import *
import numpy as np
import tensorflow as tf


class SarsaAgent():
    """
    Go back to our initial idea of training aggregated reward directly.
    No need for two net work. Just one network to fit for the aggregated reward for 
    each (s,a) pair
    """
    def __init__(self, epochs=10, length=129):
        self.epochs=epochs
        self.fit_iterations=10
        self.length = length
        # number of samples drawn every time
        self.mtrain = 1000
        self.gamma = 0.99
        self.epsilon = 0.1
        self.create_model()
        self.data = []
        self.replaybuffer = 1000000
        self.win_reward = 100
        self.reward_points_per_turn = 1

    def create_model(self):
        model = tf.keras.models.Sequential([
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
        self.model = model
        return

    def create_model_5layers(self):
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
        self.model = model
        return

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
        fit_target_network
        computes the target network prediction and fit for it with prediction network
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
    # easier to modify for different fittin and reward functions

    def compare_bots(self, bots, num_games=50):
        start_time = time.time()
        wins = {bot: 0 for bot in bots}
        final_scores = {bot:[] for bot in bots}
        for i in range(num_games):
            random.shuffle(bots)
            game = Game.setup(bots, variable_cards)
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
        game = Game.setup(players, variable_cards)
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
        if self.reward_points_per_turn:
            for p, s in scores:
                p.rewards[-1] += 100*s/k
        return scores

    def scores_to_data(self, scores, gamma = 0.99):
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
            next_states.append(player.next_states)
            ar = np.zeros_like(player.rewards,dtype=float)
            for i,r in enumerate(player.rewards[::-1]):
                ar[i] = r + gamma*ar[i-1]
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
        # only save the data vector if a card is bought (action!=0)
        select = np.sum(actions, 1) > 0
        states = states[select,:]
        actions = actions[select,:]
        rewards = rewards[select,:]
        next_states = next_states[select,:]
        aggregated_rewards = aggregated_rewards[select,:]
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
