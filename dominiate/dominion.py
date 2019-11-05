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

def compare_bots(bots, num_games=50):
    scores = {bot: 0 for bot in bots}
    for i in range(num_games):
        random.shuffle(bots)
        game = Game.setup(bots, variable_cards)
        results = game.run()
        maxscore = 0
        for bot, score in results:
            if score > maxscore: maxscore = score
        for bot, score in results:
            if score == maxscore:
                scores[bot] += 1
                break
    return scores

def test_game():
    player1 = smithyComboBot
    player2 = chapelComboBot
    player3 = HillClimbBot(2, 3, 40)
    player2.setLogLevel(logging.DEBUG)
    game = Game.setup([player1, player2, player3], variable_cards)
    results = game.run()
    return results

def human_game():
    player1 = smithyComboBot
    player2 = chapelComboBot
    player3 = HillClimbBot(2, 3, 40)
    player4 = HumanPlayer('You')
    game = Game.setup([player1, player2, player3, player4], variable_cards[-10:])
    return game.run()

def run(players):
    game = Game.setup(players, variable_cards)
    while not game.over():
        game = game.take_turn()
    scores = [(state.player, state.score()) for state in game.playerstates]
    winner, _ = max(scores, key=lambda item: item[1])
    loser, _ = min(scores, key=lambda item: item[1])
    winner.rewards[-1] += 100
    loser.rewards[-1] += -100
    return scores

def scores_to_data(scores, output_player="RLPlayer", gamma = 0.999):
    """
    output history and reward in the form of numpy array
    Outputs all player if output_player is None. Otherwise, only output for the given player.
    turn the scores output from run() to X = (m, len(data vector)) is the game state array
    and Y = (m, 1) is the reward array
    where m is the number of states that were played through in the game
    """
    Xlist = []
    Ylist = []
    for player, _ in scores:
        if output_player and player.name != output_player:
            continue
        Xlist.append(np.array(player.states))
        Y_this = np.zeros_like(player.reward)
        for i,r in enumerate(player.reward[::-1]):
            Y_this[i] = r + gamma*Y_this[i-1]
        Ylist.append(Y_this[::-1])
    X = np.concatenate(Xlist)
    Y = np.concatenate(Ylist)
    return (X,Y)

def record_game(n, players, output_player="RLPlayer", filename=''):
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
    start_time = time.time()
    for i in range(n):
        if i % 100 == 0:
          print("Playing game# %d" % i)
        # clear player history
        for p in players:
            p.reset_history()
        run(players)
        for p in players:
            if p.name == output_player:
                states.append(p.states)
                actions.append(p.actions)
                rewards.append(p.rewards)
                next_states.append(p.next_states)
    print("Took %.3f seconds" % (time.time() - start_time))
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)
    next_states = np.concatenate(next_states)
    if not filename == '':
        with open(filename, 'wb') as f:
            pickle.dump((states, actions, rewards, next_states), f)
    return states, actions, rewards, next_states

def load_game_data(filename):
    """
    load filename saved by record_game()
    """
    with open(filename, 'rb') as f:
        (X,Y) = pickle.load(f)
    return X,Y


if __name__ == '__main__':
  players = [RLPlayer(lambda x: 0), smithyComboBotFactory()]
  print(compare_bots(players))
  #record_game(1000, players, "data/smithy_vs_rl")
