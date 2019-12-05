from decision import TrashDecision, DiscardDecision
from players import AIPlayer, BigMoney
import cards as c
import logging, sys
import random

class SmithyBot(BigMoney):
    def __init__(self, cutoff1=3, cutoff2=6, cards_per_smithy=8):
        self.cards_per_smithy = 8
        self.name = 'SmithyBot(%d, %d, %d)' % (cutoff1, cutoff2,
        cards_per_smithy)
        BigMoney.__init__(self, cutoff1, cutoff2)
    
    def num_smithies(self, state):
        return list(state.all_cards()).count(c.smithy)

    def buy_priority_order(self, decision):
        state = decision.state()
        provinces_left = decision.game.card_counts[c.province]
        if provinces_left <= self.cutoff1:
            order = [None, c.estate, c.silver, c.duchy, c.province]
        elif provinces_left <= self.cutoff2:
            order = [None, c.silver, c.smithy, c.duchy, c.gold, c.province]
        else:
            order = [None, c.silver, c.smithy, c.gold, c.province]
        if ((self.num_smithies(state) + 1) * self.cards_per_smithy
           > state.deck_size()) and (c.smithy in order):
            order.remove(c.smithy)
        return order

    def make_act_decision(self, decision):
        return c.smithy

class SmithyWitchBot(BigMoney):
    def __init__(self, cutoff1=3, cutoff2=6, cards_per_smithy=8):
        self.cards_per_smithy = 8
        self.name = 'SmithyWitchBot(%d, %d, %d)' % (cutoff1, cutoff2,
        cards_per_smithy)
        BigMoney.__init__(self, cutoff1, cutoff2)
    
    def num_smithies(self, state):
        return list(state.all_cards()).count(c.smithy) + list(state.all_cards()).count(c.witch)

    def buy_priority_order(self, decision):
        state = decision.state()
        provinces_left = decision.game.card_counts[c.province]
        if provinces_left <= self.cutoff1:
            order = [None, c.estate, c.silver, c.duchy, c.province]
        elif provinces_left <= self.cutoff2:
            order = [None, c.silver, c.smithy, c.witch, c.duchy, c.gold, c.province]
        else:
            order = [None, c.silver, c.smithy, c.witch, c.gold, c.province]
        if ((self.num_smithies(state) + 1) * self.cards_per_smithy
           > state.deck_size()):
            if (c.smithy in order):
                order.remove(c.smithy)
            if (c.witch in order):
                order.remove(c.witch)
        return order

    def make_act_decision(self, decision):
        if c.witch in decision.choices():
            return c.witch
        return c.smithy

class RandomActionBot(BigMoney):
    def __init__(self, cutoff1=3, cutoff2=6, cards_per_action=8):
        self.cards_per_action = cards_per_action
        self.name = 'RandomActionBot(%d, %d, %d)' % (cutoff1, cutoff2,
        cards_per_action)
        BigMoney.__init__(self, cutoff1, cutoff2)
        self.cost2 = [cc for cc in c.variable_cards if cc.cost ==2]
        self.cost3 = [cc for cc in c.variable_cards if cc.cost ==3]
        self.cost4 = [cc for cc in c.variable_cards if cc.cost ==4]
        self.cost5 = [cc for cc in c.variable_cards if cc.cost ==5]
    
    def num_actions(self, state):
        return len([c for c in state.all_cards() if c.actions])
    def shuffle_all(self):
        random.shuffle(self.cost2)
        random.shuffle(self.cost3)
        random.shuffle(self.cost4)
        random.shuffle(self.cost5)
        return

    def buy_priority_order(self, decision):
        state = decision.state()
        provinces_left = decision.game.card_counts[c.province]
        if ((self.num_actions(state) + 1) * self.cards_per_action
           > state.deck_size()):
            if provinces_left <= self.cutoff1:
                order = [None, c.estate, c.silver, c.duchy, c.province]
            elif provinces_left <= self.cutoff2:
                order = [None, c.silver,  c.duchy, c.gold, c.province]
            else:
                order = [None, c.silver,  c.gold, c.province]
        else: 
            if provinces_left <= self.cutoff1:
                order = [None, c.estate, c.silver, c.duchy, c.province]
            elif provinces_left <= self.cutoff2:
                self.shuffle_all()
                order = [None] +  \
                      self.cost2 + \
                      self.cost3 +\
                      self.cost4+\
                      self.cost5+\
                    [c.duchy, c.gold, c.province]
            else:
                self.shuffle_all()
                order = [None] + \
                      self.cost2 +\
                      self.cost3 +\
                      self.cost4+\
                      self.cost5+\
                      [c.gold, c.province]
        return order


class SmithyCouncilBot(BigMoney):
    def __init__(self, cutoff1=3, cutoff2=6, cards_per_smithy=8):
        self.cards_per_smithy = 8
        self.name = 'SmithyCouncilBot(%d, %d, %d)' % (cutoff1, cutoff2,
        cards_per_smithy)
        BigMoney.__init__(self, cutoff1, cutoff2)
    
    def num_smithies(self, state):
        return list(state.all_cards()).count(c.smithy) + list(state.all_cards()).count(c.council_room)

    def buy_priority_order(self, decision):
        state = decision.state()
        provinces_left = decision.game.card_counts[c.province]
        if provinces_left <= self.cutoff1:
            order = [None, c.estate, c.silver, c.duchy, c.province]
        elif provinces_left <= self.cutoff2:
            order = [None, c.silver, c.smithy, c.council_room, c.duchy, c.gold, c.province]
        else:
            order = [None, c.silver, c.smithy, c.council_room, c.gold, c.province]
        if ((self.num_smithies(state) + 1) * self.cards_per_smithy
           > state.deck_size()):
            if (c.smithy in order):
                order.remove(c.smithy)
            if (c.council_room in order):
                order.remove(c.council_room)
        return order

    def make_act_decision(self, decision):
        if c.council_room in decision.choices():
            return c.council_room
        return c.smithy

class HillClimbBot(BigMoney):
    def __init__(self, cutoff1=2, cutoff2=3, simulation_steps=100):
        self.simulation_steps = simulation_steps
        if not hasattr(self, 'name'):
            self.name = 'HillClimbBot(%d, %d, %d)' % (cutoff1, cutoff2,
            simulation_steps)
        BigMoney.__init__(self, cutoff1, cutoff2)

    def buy_priority(self, decision, card):
        state = decision.state()
        total = 0
        if card is None: add = ()
        else: add = (card,)
        for coins, buys in state.simulate_hands(self.simulation_steps, add):
            total += buying_value(coins, buys)

        # gold is better than it seems
        if card == c.gold: total += self.simulation_steps/2
        self.log.debug("%s: %s" % (card, total))
        return total
    
    def make_buy_decision(self, decision):
        choices = decision.choices()
        provinces_left = decision.game.card_counts[c.province]
        
        if c.province in choices: return c.province
        if c.duchy in choices and provinces_left <= self.cutoff2:
            return c.duchy
        if c.estate in choices and provinces_left <= self.cutoff1:
            return c.estate
        return BigMoney.make_buy_decision(self, decision)

def buying_value(coins, buys):
    if coins > buys*8: coins = buys*8
    if (coins - (buys-1)*8) in (1, 7):  # there exists a useless coin
        coins -= 1
    return coins

