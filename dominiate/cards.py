import numpy as np
from decision import TrashDecision, DiscardDecision

class Card(object):
    """
    Represents a class of card.

    To save computation, only one of each card should be constructed. Decks can
    contain many references to the same Card object.
    """
    def __init__(self, name, cost, treasure=0, vp=0, coins=0, cards=0,
                 actions=0, buys=0, potionCost=0, effect=(), isAttack=False,
                 isDefense=False, reaction=(), duration=()):
        self.name = name
        self.cost = cost
        self.potionCost = potionCost
        if isinstance(treasure, int):
            self.treasure = treasure
        else:
            self.treasure = property(treasure)
        if isinstance(vp, int):
            self.vp = vp
        else:
            self.vp = property(vp)
        self.coins = coins
        self.cards = cards
        self.actions = actions
        self.buys = buys
        self._isAttack = isAttack
        self._isDefense = isDefense
        if not isinstance(effect, (tuple, list)):
            self.effect = (effect,)
        else:
            self.effect = effect
        self.reaction = reaction
        self.duration = duration

    def isVictory(self):
        return self.vp > 0

    def isCurse(self):
        return self.vp < 0

    def isTreasure(self):
        return self.treasure > 0

    def isAction(self):
        return (self.coins or self.cards or self.actions or self.buys or
                self.effect)

    def isAttack(self):
        return self._isAttack

    def isDefense(self):
        return self._isDefense

    def perform_action(self, game):
        assert self.isAction()
        if self.cards:
            game = game.current_draw_cards(self.cards)
        if (self.coins or self.actions or self.buys):
            game = game.change_current_state(
              delta_coins=self.coins,
              delta_actions=self.actions,
              delta_buys=self.buys
            )
        for action in self.effect:
            game = action(game)
        return game

    def __str__(self): return self.name
    def __eq__(self, other):
        if other is None:
             return False
        return (self.cost, self.name) == (other.cost, other.name)
    def __ne__(self, other):
        return not (self == other)
    def __lt__(self, other):
        if other is None:
            return True
        return (self.cost, self.name) < (other.cost, other.name)
    def __le__(self, other):
        return self < other or self == other
    def __gt__(self, other):
        if other is None:
            return False
        return other < self
    def __ge__(self, other):
        return self > other or self == other
    def __hash__(self):
        return hash(self.name)
    def __repr__(self): return self.name

# define the cards that are in every game
curse    = Card('Curse', 0, vp=-1)
estate   = Card('Estate', 2, vp=1)
duchy    = Card('Duchy', 5, vp=3)
province = Card('Province', 8, vp=6)

copper = Card('Copper', 0, treasure=1)
silver = Card('Silver', 3, treasure=2)
gold   = Card('Gold', 6, treasure=3)

# simple actions
village = Card('Village', 3, actions=2, cards=1)
woodcutter = Card('Woodcutter', 3, coins=2, buys=1)
smithy = Card('Smithy', 4, cards=3)
festival = Card('Festival', 5, coins=2, actions=2, buys=1)
market = Card('Market', 5, coins=1, cards=1, actions=1, buys=1)
laboratory = Card('Laboratory', 5, cards=2, actions=1)

def chapel_action(game):
    newgame = game.current_player().make_decision(
        TrashDecision(game, 0, 4)
    )
    return newgame

def cellar_action(game):
    newgame = game.current_player().make_decision(
        DiscardDecision(game)
    )
    card_diff = game.state().hand_size() - newgame.state().hand_size()
    return newgame.replace_current_state(newgame.state().draw(card_diff))

def warehouse_action(game):
    newgame = game.current_player().make_decision(
        DiscardDecision(game, 3, 3)
    )
    return newgame

def council_room_action(game):
    return game.change_other_states(delta_cards=1)

def militia_attack(game):
    return game.attack_with_decision(
        lambda g: DiscardDecision(g, 2, 2)
    )

chapel = Card('Chapel', 2, effect=chapel_action)
cellar = Card('Cellar', 2, actions=1, effect=cellar_action)
warehouse = Card('Warehouse', 3, cards=3, actions=1, effect=warehouse_action)
council_room = Card('Council Room', 5, cards=4, buys=1,
                    effect=council_room_action)
militia = Card('Militia', 4, coins=2, effect=militia_attack)
moat = Card('Moat', 2, cards=2, isDefense=True)

"""
# for now limit cards to only points plus treasure
variable_cards = []
CARD_VECTOR_ORDER = (
# Points
estate, duchy, province,
# Treasures
copper, silver, gold)
"""


variable_cards = [village, cellar, smithy, festival, market, laboratory,
chapel, warehouse, council_room, militia, moat]
CARD_VECTOR_ORDER = (
# Points
curse, estate, duchy, province,
# Treasures
copper, silver, gold,
# Actions
village, cellar, smithy, festival, market, laboratory, chapel, warehouse, council_room, militia, moat)

CARD_TO_INDEX = {c : i for i, c in enumerate(CARD_VECTOR_ORDER)}

def card_to_vector(card):
    vec = np.zeros(len(CARD_VECTOR_ORDER))
    if card:
        vec[CARD_TO_INDEX[card]] = 1
    return vec
