from game import *
from players import *
from basic_ai import *
from combobot import *
from cards import variable_cards
from collections import defaultdict
from rl_agent import RLPlayer
import pickle
import random
import time
from sarsa_trainer import SarsaAgent
import tensorflow as tf

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
    
def load_rl_bot(fn, dql=''):
    if dql == '':
        # has not initialize SarsaAgent's model network, start a new one
        dql = SarsaAgent()
        dql.create_model_5layers()
        data = dql.load_game_data('1game')
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


    
    

def test_game():
    player1 = smithyComboBot
    player2 = chapelComboBot
    player3 = HillClimbBot(2, 3, 40)
    player2.setLogLevel(logging.DEBUG)
    game = Game.setup([player1, player2, player3], variable_cards)
    results = game.run()
    return results

def play_rlbot(fname='model_upload/combination_v0_iteration_999'):
    # 764 also quite strong
    player1 = HumanPlayer('You')
    # initialize the network
    dql = SarsaAgent()
    dql.create_model_5layers()
    data = dql.load_game_data('1game')
    dql.fit(data)
    dql.load_model(fname)
    player2 = RLPlayer(lambda x: dql.model.predict(x))
    player2.epsilon=0
    player2.setLogLevel(logging.INFO)
    game = Game.setup([player1, player2], variable_cards,False)
    return game.run()

def human_game():
    #player1 = smithyComboBot
    #player2 = chapelComboBot
    #player3 = HillClimbBot(2, 3, 40)
    player1 = SmithyBot()
    player4 = HumanPlayer('You')
    #game = Game.setup([player1, player2, player3, player4], variable_cards[-10:])
    game = Game.setup([player1, player4], variable_cards,False)
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
        next_states.append(player.next_states)
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



def load_game_data(filename):
    """
    load filename saved by record_game()
    """
    with open(filename, 'rb') as f:
        return  pickle.load(f)


if __name__ == '__main__':
  players = [RLPlayer(lambda x: 0), smithyComboBotFactory()]
  print(compare_bots(players))
  #record_game(1000, players, "data/smithy_vs_rl")
