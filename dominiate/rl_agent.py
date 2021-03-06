import random
import numpy as np
import cards as c

from players import AIPlayer


class BuyPolicyPlayer(AIPlayer):

  def __init__(self):
    AIPlayer.__init__(self)

  def make_buy_decision(self, decision):
    raise NotImplementedError("BuyPolicyPlayer is an abstract class.")

  def act_priority(self, decision, choice):
    """
        Assign a numerical priority to each action. Higher priority actions
        will be chosen first.
        """
    if choice is None:
      return 0
    return (100 * choice.actions + 10 *
            (choice.coins + choice.cards) + choice.buys) + 1

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
          decision, choices, allow_none=(len(chosen) >= decision.min))
      if latest is not None:
        choices.remove(latest)
        chosen.append(latest)
    return chosen

  def make_discard_decision_incremental(self,
                                        decision,
                                        choices,
                                        allow_none=True):
    actions_sorted = [ca for ca in choices if ca.isAction()]
    actions_sorted.sort(key=lambda a: a.actions)
    plus_actions = sum([ca.actions for ca in actions_sorted])
    wasted_actions = len(
        actions_sorted) - plus_actions - decision.state().actions
    victory_cards = [
        ca for ca in choices
        if ca.isVictory() and not ca.isAction() and not ca.isTreasure()
    ]
    if wasted_actions > 0:
      return actions_sorted[0]
    elif len(victory_cards):
      return victory_cards[0]
    elif c.copper in choices:
      return c.copper
    elif allow_none:
      return None
    else:
      priority_order = sorted(
          choices, key=lambda ca: (ca.actions, ca.cards, ca.coins, ca.treasure))
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
          decision, choices, allow_none=(len(chosen) >= decision.min))
      if latest is not None:
        choices.remove(latest)
        chosen.append(latest)
    return chosen


class RandomPlayer(BuyPolicyPlayer):

  def __init__(self):
    self.name = "RandomPlayer"
    BuyPolicyPlayer.__init__(self)

  def make_buy_decision(self, decision):
    choices = decision.choices()
    return np.random.choice(choices)


class RandomActPlayer(BuyPolicyPlayer):
  """
    Buy cards and play actions randomly
    """

  def __init__(self):
    self.name = "RandomActPlayer"
    BuyPolicyPlayer.__init__(self)

  def make_buy_decision(self, decision):
    choices = decision.choices()
    return np.random.choice(choices)

  def make_act_decision(self, decision):
    choices = decision.choices()
    return np.random.choice(choices)


class RLPlayer(BuyPolicyPlayer):

  def __init__(self, value_fn, epsilon=0.1, include_action=1):
    self.value_fn = value_fn
    # if input is (s,a): include_action = 1
    # if input is (s): include_action = 0
    self.include_action = include_action
    self.name = "RLPlayer"
    self.epsilon = epsilon
    BuyPolicyPlayer.__init__(self)

  def advantage(self, decision):
    choices = decision.choices()
    choices_vec = np.zeros((len(choices), len(c.CARD_VECTOR_ORDER)))
    if not choices:
      return choices
    for i, choice in enumerate(choices):
      choices_vec[i, :] = c.card_to_vector(choice)
    state = np.tile(np.array(decision.game.to_vector()), (len(choices), 1))
    if self.include_action:
      X = np.concatenate((state, choices_vec), axis=1)
      return self.value_fn(X)
    else:
      X = np.array([
          np.array(decision.choose(choice, simulate=True).to_vector())
          for choice in choices
      ])
      rewards = np.array([choice.vp if choice else 0 for choice in choices
                         ]).reshape([-1, 1])
      advantage = self.value_fn(X) + rewards
      return advantage

  def make_buy_decision(self, decision):
    """Chooses a card to buy.

    By default, this chooses the card with the highest positive
    buy_priority.
    """
    choices = decision.choices()
    # Random exploration with probability epsilon.
    if (np.random.random() < self.epsilon):
      return np.random.choice(choices)
    weights = self.advantage(decision)
    return choices[np.argmax(weights)]

  def act_priority(self, decision, choice):
    """Assigns numerical priority to each action.

    Higher card will be played first.
    """
    if choice is None:
      return 0
    return (100 * choice.actions + 20 * choice._isAttack + 15 * choice.cards +
            10 * choice.coins + choice.buys) + 1

  def make_act_decision(self, decision):
    """
        Choose an Action to play.

        By default, this chooses the action with the highest positive
        act_priority.
        """
    choices = decision.choices()
    choices.sort(key=lambda x: self.act_priority(decision, x))
    return choices[-1]


# create a class for a RL bot with both buy network and act network
class BuyActRLplayer(RLPlayer):
  """RL agent that also learns the priority of playing action cards."""

  def __init__(self,
               value_fn,
               act_fn,
               epsilon=0.1,
               epsilon_act=0.1,
               include_action=1):
    RLPlayer.__init__(self, value_fn, epsilon, include_action)
    self.value_fn = value_fn
    self.name = "BuyActRLPlayer"
    self.act_fn = act_fn
    self.epsilon_act = epsilon_act

  def make_act_decision(self, decision):
    """ Choose an Action to play.

    use the function out put of act_fn to choos the function to play
    """
    choices = decision.choices()
    # Random exploration with probability epsilon.
    if (np.random.random() < self.epsilon_act):
      return np.random.choice(choices)
    weights = self.act_advantage(decision)
    return choices[np.argmax(weights)]

  def act_advantage(self, decision):
    choices = decision.choices()
    choices_vec = np.zeros((len(choices), len(c.CARD_VECTOR_ORDER)))
    if not choices:
      return choices
    for i, choice in enumerate(choices):
      choices_vec[i, :] = c.card_to_vector(choice)
    state = np.tile(np.array(decision.game.to_vector()), (len(choices), 1))
    X = np.concatenate((state, choices_vec), axis=1)
    return self.act_fn(X)
