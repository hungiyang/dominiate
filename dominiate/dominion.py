import numpy as np
from game import Game
from players import HumanPlayer
from basic_ai import SmithyBot
from cards import variable_cards
from collections import defaultdict
from rl_agent import RLPlayer, RandomPlayer, BuyActRLplayer
import pickle
import random
import time
import h5py
from dqltrainer import DQLSarsaAgent
from sarsa_trainer import SarsaAgent, SarsaActBuyAgent
#import tensorflow as tf

def compare_bots(bots, num_games=50, order = 0):
    """
    Compare the performance of the two bots
    Since when winrate is close, who goes first matters a lot, 
    I'll add that to the comparison
    """
    start_time = time.time()
    wins = {bot: 0 for bot in bots}
    final_scores = {bot:[] for bot in bots}
    for i in range(num_games):
        # random.shuffle(bots)
        game = Game.setup(bots, variable_cards)
        if order:
            if game.playerstates[0].player != bots[0]:
                game.playerstates.reverse()
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

def compare_rl_bots(fn1, fn2, num_games = 50, order = 0):
    # set up the two rl bots, and make them fight.
    print('setting up bot1...')
    dql = SarsaAgent()
    dql.create_model_5layers()
    data = dql.load_game_data('1game')
    dql.fit(data)
    dql.load_model(fn1)
    p1 = RLPlayer(lambda x: dql.model.predict(x))
    p1.name = fn1
    p1.epsilon = 0
    print('setting up bot2...')
    dql2 = SarsaAgent()
    dql2.create_model_5layers()
    data = dql2.load_game_data('1game')
    dql2.fit(data)
    dql2.load_model(fn1)
    p2 = RLPlayer(lambda x: dql2.model.predict(x))
    p2.name = fn2
    p2.epsilon = 0
    print('fight!')
    return compare_bots([p1, p2], num_games, order)
    
"""
def load_rl_bot(fn, dql='', pre_witch=0, actbot = 0):
    if dql == '':
        if actbot:
            pr = RandomPlayer()
            pr.record_history=1
            dql = SarsaActBuyAgent()
            data = dql.record_game(1, [pr, SmithyBot()],verbose=0)
            dql.length=(data[0].shape[1]+data[1].shape[1])
            dql.create_model_5layers()
            dql.create_act_model()
            dql.fit(data)
            dql.fit_act(data)
            dql.load_model(fn)
            p1 = BuyActRLplayer(lambda x: dql.model.predict(x), lambda x: dql.model_act.predict(x))
            p1.name = fn
        else:
            # has not initialize SarsaAgent's model network, start a new one
            if pre_witch:
                data = load_game_data('1game')
            else:
                pr = RandomPlayer()
                pr.record_history=1
                data = record_game(1, [pr, SmithyBot()])
            dql = SarsaAgent(length=(data[0].shape[1]+data[1].shape[1]))
            dql.create_model_5layers()
            dql.fit(data)
            dql.load_model(fn)
            p1 = RLPlayer(lambda x: dql.model.predict(x))
            p1.name = fn
    else:
        dql.load_model(fn)
        p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.name = fn
    p1.epsilon = 0
    return (p1, dql)
"""

def get_num_layers(fn):
    """
    get the number of layers of a .h5 weight file
    """
    with h5py.File(fn, 'r') as f:
        # List all groups
        all_key = list(f.keys())
    return len([k for k in all_key if 'dropout' in k])	


def load_rl_bot(fn, dql='', version = 'SarsaActBuyAgent', pre_witch = 0):
    if dql == '':
        if version == 'SarsaActBuyAgent':
            pr = RandomPlayer()
            pr.record_history=1
            dql = SarsaActBuyAgent()
            data = dql.record_game(1, [pr, SmithyBot()],verbose=0)
            dql.length=(data[0].shape[1]+data[1].shape[1])
            dql.create_model_5layers()
            dql.create_act_model()
            dql.fit(data)
            dql.fit_act(data)
            dql.load_model(fn)
            p1 = BuyActRLplayer(lambda x: dql.model.predict(x), lambda x: dql.model_act.predict(x))
            p1.name = fn
        elif version == 'SarsaAgent':
            if pre_witch:
                variable_cards_this = [
                    village, cellar, smithy, festival, market, laboratory, chapel, warehouse,
                    council_room, militia, moat ]
                pr = RandomPlayer()
                pr.record_history=1
                dql = SarsaAgent()
                dql.variable_cards = variable_cards_this
                data = dql.record_game(1, [pr, SmithyBot()])
                print(data[0].shape)
                print(data[1].shape)
            else:
                pr = RandomPlayer()
                pr.record_history=1
                data = record_game(1, [pr, SmithyBot()])
            dql = SarsaAgent(length=(data[0].shape[1]+data[1].shape[1]))
            dql.create_model(num_layers = get_num_layers(fn + '_ar.h5'))
            dql.fit(data)
            dql.load_model(fn)
            p1 = RLPlayer(lambda x: dql.model.predict(x))
            p1.name = fn
        elif version == 'DQLSarsaAgent':
            dql = DQLSarsaAgent()
            p1 = SmithyBot()
            p1.record_history = 1
            dbuy, _ = dql.record_game(1, [p1,SmithyBot()])
            sa = np.array([np.concatenate([s,a]) for s,a,r,_,_,_ in dbuy])
            r = np.array([r for s,a,r,_,_,_ in dbuy])
            dql.create_model(sa, r)
            dql.load_model(fn)
            p1 = RLPlayer(lambda x: dql.model_predict.predict(x))
            p1.name = fn
        else:
            print('No such version of Agent!')
            raise 
    else:
        # if dql is NN is already initialized.
        dql.load_model(fn)
        if version == 'DQLSarsaAgent':
            p1 = RLPlayer(lambda x: dql.model_predict.predict(x))
        else:   
            p1 = RLPlayer(lambda x: dql.model.predict(x))
        p1.name = fn
    p1.epsilon = 0
    return (p1, dql)
    

def play_rlbot(fname='model_upload/BuyActRL_v2_iteration_999', dql='', version = 'SarsaActBuyAgent', pre_witch=0):
    player1 = HumanPlayer('You')
    player2, dql = load_rl_bot(fname,dql,version, pre_witch)
    game = Game.setup([player1, player2], variable_cards,False)
    game.run()
    return dql

def human_game(p = []):
    if p == []:
        p = SmithyBot()
    ready_player_one = HumanPlayer('You')
    game = Game.setup([p, ready_player_one], variable_cards,False)
    return game.run()

def run(players, win_reward = 0):
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
    winner.rewards[-1] += win_reward
    loser.rewards[-1] += -win_reward
    return scores

def scores_to_data(scores, gamma = 0.99):
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
        next_states.append(player.choices)
        ar = np.zeros_like(player.rewards,dtype=float)
        for i,r in enumerate(player.rewards[::-1]):
            ar[i] = r + gamma*ar[i-1]
        aggregated_rewards.append(ar[::-1])
    final_score = {bot: fs for bot, fs in scores}
    
    return np.concatenate(states), np.concatenate(actions), np.concatenate(rewards), np.concatenate(next_states),np.concatenate(aggregated_rewards), final_score

def record_game(n, players, filename='', win_reward=0, verbose=1):
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
        s, a, r, n, ar, fs = scores_to_data(run(players, win_reward))
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
    ## only save the data vector if a card is bought (action!=0) --> why did I need this? A bad idea...
    #select = np.sum(actions, 1) > 0
    #states = states[select,:]
    #actions = actions[select,:]
    #rewards = rewards[select,:]
    #next_states = next_states[select,:]
    #aggregated_rewards = aggregated_rewards[select,:]
    if not filename == '':
        with open(filename, 'wb') as f:
            pickle.dump((states, actions, rewards, next_states, aggregated_rewards), f)
    return states, actions, rewards, next_states, aggregated_rewards



def load_game_data(filename):
    """
    load filename saved by record_game()
    """
    with open(filename, 'rb') as f:
        return  pickle.load(f)


if __name__ == '__main__':
  play_rlbot()
