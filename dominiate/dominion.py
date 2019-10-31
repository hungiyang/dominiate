from game import *
from players import *
from basic_ai import *
from combobot import *
from cards import variable_cards
from collections import defaultdict
import random
import pickle

def compare_bots(bots):
    scores = defaultdict(int)
    for i in xrange(50):
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

def run():
    """
    Play a game of Dominion. Return a dictionary mapping players to scores.
    """
    player1 = smithyComboBot
    player2 = chapelComboBot
    # player3 = HillClimbBot(2, 3, 40)
    game = Game.setup([player1, player2], variable_cards)
    while not game.over():
        game = game.take_turn()
    scores = [(state.player, state.score()) for state in game.playerstates]
    winner, _ = max(scores, key=lambda item: item[1])
    loser, _ = min(scores, key=lambda item: item[1])
    winner.reward[-1] += 100
    loser.reward[-1] += -100

    return scores

def scores_to_data(scores, player = 0):
    """
    output player 0's history and reward in the form of numpy array
    turn the scores output from run() to X = (m, len(data vector)) is the game state array
    and Y = (m, 1) is the reward array
    where m is the number of samples. (number of states)
    """
    X = np.array(scores[player][0].history)
    Y = np.array(scores[player][0].reward)

    return (X,Y)

def record_game(n, filename):
    """
    play n games and save the results in filename
    save tuple (X,Y)
    X has size (m, len(data vector))
    Y has size (m, 1)
    m is the number of game states recorded
    """
    X = []
    Y = []
    for i in xrange(10):
        xtmp, ytmp = scores_to_data(run())
        X.append(xtmp)
        Y.append(ytmp)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    # save X, Y to filename
    with open(filename, 'wb') as f:
        pickle.dump((X,Y), f)

    return 

def load_game_data(filename):
    """
    load filename saved by record_game()
    """
    with open(filename, 'rb') as f:
        (X,Y) = pickle.load(f)
    return X,Y


if __name__ == '__main__':
    #print compare_bots([smithyComboBot, chapelComboBot, HillClimbBot(2, 3, 40)])
    human_game()
