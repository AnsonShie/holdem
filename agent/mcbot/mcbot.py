from random import sample

from treys import Card
from treys import Deck
from treys import Evaluator

import json
from multiprocessing.pool import ThreadPool
import time


class PseudoDeck:
    def __init__(self, deck):
        self._cards = list(deck)

    def draw(self, n=2):
        return sample(self._cards, n)

#class CalcBot:
#    def estimate_winrate(self, card_strs):
#        cal_strength = card_strength(card_strs, [])
        #print cal_strength
#        return cal_strength

class PreflopBot:
    """
    Assign winning odds using pre-defined rates.
    """
    def __init__(self):
        self._lookup = json.load(open("data/preflop_odds.json"))

    def estimate_winrate(self, card_strs):
        key = self.__construct_preflop_key(card_strs)
        if key in self._lookup.keys():
            odds = float(self._lookup[key]["win_odds"])
        else:
            odds = float(self._lookup[key[:-1]]["win_odds"])
        return odds / 100

    def __construct_preflop_key(self, card_strs):
        cards = sorted(Card.hand_to_binary(card_strs), reverse=True)
        #print Card.print_pretty_cards(cards)
        key = ''.join([Card.int_to_str(card)[0] for card in cards])
        if Card.get_suit_int(cards[0]) == Card.get_suit_int(cards[1]):
            key = key + "s"
        else:
            key = key + "o"
        return key
    

class MonteCarloBot:
    """
    Perform 1-to-1 poker winning odds simulation with personal and community cards.

    internal states
        _deck: deck of card on dealer's hand (max 52)
        _evaluator: poker hand evaluator
        _board: community cards drawed on board
        _hand: personal cards drawed to hand
    Usage flow
        1. init
        2. pre-flop stage: draw personal cards from _deck (self.assign_hand)
        3. turn stages: draw community cards (self.assign_board)
            3a. estimate winning rate: run Monte Carlo simulation (self.estimate_winrate)
            3b. loop to 3. to assign another community cards, if needed.
        4. game over or round end: reset the internal deck to 52 cards (self.reset)
    """
    _sim_n = 100

    def __init__(self):
        self._deck = Deck()
        self._evaluator = Evaluator()
        self._board = list()
        self._hand = list()

    def reset(self):
        self.__init__()

    def assign_hand(self, card_strs):
        # card_strs suit must be lower case: ['Ac', 'Ts']
        if len(self._hand) == 0:
            for card in card_strs:
                self._hand.append(Card.new(card))
            self._remove_from_deck(card_strs)

    def assign_board(self, card_strs):
        # card_strs suit must be lower case: ['Ah', 'Js', '7h']
        for card in card_strs:
            if Card.hand_to_binary([card])[0] not in self._board:
                self._board.append(Card.new(card))
        self._remove_from_deck(card_strs)

    def estimate_winrate(self, n=_sim_n):
        #print self._hand, self._board
        pd = PseudoDeck(self._deck.cards)
        cur_rank = self._evaluator.evaluate(self._hand, self._board)
        #print 'Rank: {0}'.format(cur_rank)
        wincount = sum([self._simulation(pd, self._board, cur_rank) for _ in range(n)])
        return wincount / n

    def _simulation(self, deck, board, myrank):
        hand = deck.draw(2)
        simrank = self._evaluator.evaluate(hand, board)
        return 1.0 if myrank < simrank else 0.0

    def _remove_from_deck(self, card_strs):
        cards = Card.hand_to_binary(card_strs)
        for card in cards:
            if card in self._deck.cards:
                self._deck.cards.remove(card)

        # print 'Unused cards: {0}'.format(len(self._deck.cards))


class ParaMonteCarloBot(MonteCarloBot):

    _sim_n = 20000

    def __init__(self, thread_num):
        MonteCarloBot.__init__(self)
        self.thread_num = thread_num
        self.pool = ThreadPool(thread_num)

    def reset(self):
        self.__init__(self.thread_num)

    def worker(self, task):
        result = self._simulation(*task)
        return result

    def estimate_winrate(self, n=_sim_n):
        print(self._hand, self._board)
        pd = PseudoDeck(self._deck.cards)
        cur_rank = self._evaluator.evaluate(self._hand, self._board)
        print('Rank: {0}'.format(cur_rank))
        tasks = [(pd, self._board, cur_rank) for _ in range(n)]
        start_time = time.time()
        results = self.pool.map(self.worker, tasks)
        print('calculate %d times by %d threads in %f secs' % (len(results), self.thread_num, time.time() - start_time))
        wincount = sum(results)

        return wincount / len(results)

