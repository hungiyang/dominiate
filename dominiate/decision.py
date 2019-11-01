INF = float('Inf')

class Decision(object):
    def __init__(self, game):
        self.game = game
    def state(self):
        return self.game.state()
    def player(self):
        return self.game.current_player()

class MultiDecision(Decision):
    def __init__(self, game, min=0, max=INF):
        self.min=min
        self.max=max
        Decision.__init__(self, game)

class ActDecision(Decision):
    def choices(self):
        return [None] + [card for card in self.state().hand if card.isAction()]
    def choose(self, card):
        self.game.log.info("%s plays %s" % (self.player().name, card))
        if card is None:
            newgame = self.game.change_current_state(
              delta_actions=-self.state().actions
            )
            return newgame
        else:
            newgame = card.perform_action(self.game.current_play_action(card))
            return newgame
    def __str__(self):
        return "ActDecision (%d actions, %d buys, +%d coins)" %\
          (self.state().actions, self.state().buys, self.state().coins)

class BuyDecision(Decision):
    def coins(self):
        return self.state().hand_value()
    def buys(self):
        return self.state().buys
    def choices(self):
        assert self.coins() >= 0
        value = self.coins()
        return [None] + [card for card in self.game.card_choices() if card.cost <= value]
    def choose(self, card, simulate=False):
        if not simulate:
            self.game.log.info("%s buys %s" % (self.player().name, card))
        state = self.state()
        if card is None:
            newgame = self.game.change_current_state(
              delta_buys=-state.buys
            )
            return newgame
        else:
            newgame = self.game.remove_card(card).replace_current_state(
              state.gain(card).change(delta_buys=-1, delta_coins=-card.cost)
            )
            return newgame
    
    def __str__(self):
        return "BuyDecision (%d buys, %d coins)" %\
          (self.buys(), self.coins())

class TrashDecision(MultiDecision):
    def choices(self):
        return sorted(list(self.state().hand))

    def choose(self, choices):
        self.game.log.info("%s trashes %s" % (self.player().name, choices))
        state = self.state()
        for card in choices:
            state = state.trash_card(card)
        return self.game.replace_current_state(state)

    def __str__(self):
        return "TrashDecision(%s, %s, %s)" % (self.state().hand, self.min, self.max)

class DiscardDecision(MultiDecision):
    def choices(self):
        return sorted(list(self.state().hand))
    
    def choose(self, choices):
        self.game.log.info("%s discards %s" % (self.player().name, choices))
        state = self.state()
        for card in choices:
            state = state.discard_card(card)
        return self.game.replace_current_state(state)
    
    def __str__(self):
        return "DiscardDecision" + str(self.state().hand)
