import numpy as np
from game import Game
from decision import BuyDecision, ActDecision, TrashDecision, DiscardDecision, MultiDecision, INF
import cards as c
import logging

class Player(object):
    def __init__(self, *args):
        raise NotImplementedError("Player is an abstract class")
    def make_decision(self, decision, state):
        assert state.player is self
        raise NotImplementedError
    def make_multi_decision(self, decision, state):
        raise NotImplementedError
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Player: %s>" % self.name
    def before_turn(self, game):
        pass
    def after_turn(self, game):
        pass

class HumanPlayer(Player):
    def __init__(self, name):
        self.name = name
        # add this so that code can run 
        self.states_act = []
        self.actions_act = []
        self.rewards_act = []

    def make_decision(self, decision):
        if decision.game.simulated:
            # Don't ask the player to tell the AI what they'll do!
            return self.substitute_ai().make_decision(decision)
        state = decision.game.state()
        allcards = state.hand + state.discard + state.tableau + state.drawpile
        print("Deck: %d cards" % state.deck_size())
        print("whole deck:")
        print(dict(zip(*np.unique(allcards, return_counts=True))))
        print("hand:")
        print(state.hand)
        print("VP: %d" % state.score())
        print("Opponent VP: %d" % decision.game.playerstates[1-decision.game.player_turn].score())
        print(decision)
        if isinstance(decision, MultiDecision):
            chosen = self.make_multi_decision(decision)
        else:
            chosen = self.make_single_decision(decision)
        return decision.choose(chosen)

    def make_single_decision(self, decision):
        print('? for a description of the cards.')
        for index, choice in enumerate(decision.choices()):
            if choice != None:
                print("\t[%d] %s (%d left)" % (index, choice, decision.game.card_counts[choice]))
            else:
                print("\t[%d] %s " % (index, choice))
        choice = input('Your choice: ')
        try:
            return decision.choices()[int(choice)]
        except (ValueError, IndexError):
            if choice == '?':
                print('moat(cost 2): +2 card, defense')
                print('cellar(cost 2): +1 action, discard n cards, draw n cards')
                print('chapel(cost 2): trash up to 4 cards')
                print('village(cost 3): +1 card, +2 actions')
                print('warehouse(cost 3): +1 action, +3 card, discard 3 card')
                print('smithy(cost 4): +3 cards')
                print('militia(cost 4): +2 coin, opponents discard 2 cards')
                print('festival(cost 5): +2 action, +2 coin, +1 buy')
                print('market(cost 5): +1 action, +1 card, +1 coin, +1 buy')
                print('laboratory(cost 5): +1 action, +2 card')
                print('council_room(cost 5): +4 cards, +1 buy, opponents +1 card')
                print('witch(cost 5): +2 card, opponents gain curse')
            else: 
                # Try again
                print("That's not a choice.")
            return self.make_single_decision(decision)

    def make_multi_decision(self, decision):
        for index, choice in enumerate(decision.choices()):
            print("\t[%d] %s" % (index, choice))
        if decision.min != 0:
            print("Choose at least %d options." % decision.min)
        if decision.max != INF:
            print("Choose at most %d options." % decision.max)
        choices = input('Your choices (separated by commas): ')
        try:
            chosen = [decision.choices()[int(choice.strip())]
                      for choice in choices.split(',')]
        except (ValueError, IndexError):
            # Try again
            print("That's not a valid list of choices.")
            return self.make_multi_decision(decision)
        if len(chosen) < decision.min:
            print("You didn't choose enough things.")
            return self.make_multi_decision(decision)
        if len(chosen) > decision.max:
            print("You chose too many things.")
            return self.make_multi_decision(decision)
        for ch in chosen:
            if chosen.count(ch) > 1:
                print("You can't choose the same thing twice.")
                return self.make_multi_decision(decision)
            return chosen
    
    def substitute_ai(self):
        return BigMoney()


class AIPlayer(Player):
    def __init__(self):
        self.log = logging.getLogger(self.name)
        self.record_history = False
        self.reset_history()
    def reset_history(self):
        #### reward for buy network is more straight forward
        #### just use the victory points as reward when you buy a card
        # s_t in RL.
        self.states = []
        # a_t
        self.actions = []
        # possible action at each stage
        self.choices = []
        # r(s_t, a_t).
        self.rewards = []
        ###### data recording for action phase training
        # it is tricky to set up the proper reward for training action playing
        # I think a good way is to measure the increase in playerstate.hand_value()
        # the increase in hand_value cannot be seen at make_decision though. 
        # write the states and rewards in the decision class.
        # s_t in RL.
        self.states_act = []
        # a_t
        self.actions_act = []
        # possible action choices
        self.choices_act = []
        # use final vp - current vp as the Q(s,a) 
        self.vp = []
    def setLogLevel(self, level):
        self.log.setLevel(level)
    def make_decision(self, decision):
        self.log.debug("Decision: %s" % decision)
        if isinstance(decision, BuyDecision):
            #decision.game.log.info(str(decision)) # print number of coins and buys
            choice = self.make_buy_decision(decision)
            r = 0
            if isinstance(choice, c.Card):
                r = choice.vp
            self.states.append(decision.game.to_vector())
            self.actions.append(c.card_to_vector(choice))
            #self.next_states.append(decision.choose(choice, True).to_vector())
            self.choices.append(c.cardlist_to_vector(decision.choices()))
            self.rewards.append(r)
        elif isinstance(decision, ActDecision):
            #decision.game.log.info(str(decision)) # print number of actions and coins
            decision.game.log.info(decision.choices())
            choice = self.make_act_decision(decision)
            self.states_act.append(decision.game.to_vector())
            self.actions_act.append(c.card_to_vector(choice))
            self.choices_act.append(c.cardlist_to_vector(decision.choices()))
            self.vp.append(decision.state().score())
        elif isinstance(decision, DiscardDecision):
            choice = self.make_discard_decision(decision)
        elif isinstance(decision, TrashDecision):
            choice = self.make_trash_decision(decision)
        else:
            raise NotImplementedError

        return decision.choose(choice)

class BigMoney(AIPlayer):
    """
    This AI strategy provides reasonable defaults for many AIs. On its own,
    it aims to buy money, and then buy victory (the "Big Money" strategy).
    """
    def __init__(self, cutoff1=3, cutoff2=6):
        self.cutoff1 = cutoff1  # when to buy duchy instead of gold
        self.cutoff2 = cutoff2  # when to buy duchy instead of silver
        #FIXME: names are implemented all wrong
        if not hasattr(self, 'name'):
            self.name = 'BigMoney(%d, %d)' % (self.cutoff1, self.cutoff2)
        AIPlayer.__init__(self)
    
    def buy_priority_order(self, decision):
        """
        Provide a buy_priority by ordering the cards from least to most
        important.
        """
        provinces_left = decision.game.card_counts[c.province]
        if provinces_left <= self.cutoff1:
            return [None, c.estate, c.silver, c.duchy, c.province]
        elif provinces_left <= self.cutoff2:
            return [None, c.silver, c.duchy, c.gold, c.province]
        else:
            return [None, c.silver, c.gold, c.province]
    
    def buy_priority(self, decision, card):
        """
        Assign a numerical priority to each card that can be bought.
        """
        try:
            return self.buy_priority_order(decision).index(card)
        except ValueError:
            return -1
    
    def make_buy_decision(self, decision):
        """
        Choose a card to buy.

        By default, this chooses the card with the highest positive
        buy_priority.
        """
        choices = decision.choices()
        choices.sort(key=lambda x: self.buy_priority(decision, x))
        return choices[-1]
    
    def act_priority(self, decision, choice):
        """
        Assign a numerical priority to each action. Higher priority actions
        will be chosen first.
        """
        if choice is None: return 0
        return (100*choice.actions + 10*(choice.coins + choice.cards) +
                    choice.buys) + 1
    
    def make_act_decision(self, decision):
        """
        Choose an Action to play.

        By default, this chooses the action with the highest positive
        act_priority.
        """
        choices = decision.choices()
        choices.sort(key=lambda x: self.act_priority(decision, x))
        return choices[-1]
    
    def make_trash_decision_incremental(self, decision, choices, allow_none=True):
        "Choose a single card to trash."
        deck = decision.state().all_cards()
        money = sum([card.treasure + card.coins for card in deck])
        if c.curse in choices:
            return c.curse
        elif c.copper in choices and money > 3:
            return c.copper
        elif decision.game.round < 10 and c.estate in choices:
            # TODO: judge how many turns are left in the game and whether
            # an estate is worth it
            return c.estate
        elif allow_none:
            return None
        else:
            # oh shit, we don't know what to trash
            # get rid of whatever looks like it's worth the least
            choices.sort(key=lambda x: (x.vp, x.cost))
            return choices[0]

    def make_trash_decision(self, decision):
        """
        The default way to decide which cards to trash is to repeatedly
        choose one card to trash until None is chosen.

        TrashDecision is a MultiDecision, so return a list.
        """
        latest = False
        chosen = []
        choices = decision.choices()
        while choices and latest is not None and len(chosen) < decision.max:
            latest = self.make_trash_decision_incremental(
                decision, choices,
                allow_none = (len(chosen) >= decision.min)
            )
            if latest is not None:
                choices.remove(latest)
                chosen.append(latest)
        return chosen

    def make_discard_decision_incremental(self, decision, choices, allow_none=True):
        actions_sorted = [ca for ca in choices if ca.isAction()]
        actions_sorted.sort(key=lambda a: a.actions)
        plus_actions = sum([ca.actions for ca in actions_sorted])
        wasted_actions = len(actions_sorted) - plus_actions - decision.state().actions
        victory_cards = [ca for ca in choices if ca.isVictory() and
                         not ca.isAction() and not ca.isTreasure()]
        if wasted_actions > 0:
            return actions_sorted[0]
        elif len(victory_cards):
            return victory_cards[0]
        elif c.copper in choices:
            return c.copper
        elif allow_none:
            return None
        else:
            priority_order = sorted(choices,
              key=lambda ca: (ca.actions, ca.cards, ca.coins, ca.treasure))
            return priority_order[0]

    def make_discard_decision(self, decision):
        # TODO: make this good.
        # This probably involves finding all distinct sets of cards to discard,
        # of size decision.min to decision.max, and figuring out how well the
        # rest of your hand plays out (including things like the Cellar bonus).

        # Start with 
        #   game = decision.game().simulated_copy() ...
        # to avoid cheating.

        latest = False
        chosen = []
        choices = decision.choices()
        while choices and latest is not None and len(chosen) < decision.max:
            latest = self.make_discard_decision_incremental(
                decision, choices,
                allow_none = (len(chosen) >= decision.min)
            )
            if latest is not None:
                choices.remove(latest)
                chosen.append(latest)
        return chosen

