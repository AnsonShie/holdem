from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
from agent.mcbot.mcbot import MonteCarloBot
import random

class mcModel():
    bot = MonteCarloBot()
    #preflop = CalcBot()

    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}
        self.reset_state()
        
    def reset_state(self):
        self._roundRaiseCount = 0

    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def takeAction(self, state, playerid):
        my_card = state.player_states[playerid].hand
        my_card = [card_to_normal_str(card) for card in my_card]
        community_card = state.community_card
        community_card = [card_to_normal_str(card) for card in community_card if card != -1]
        self.bot.reset()

        self.bot.assign_hand(my_card)

        if len(community_card) > 0:
            # get board cards
            self.bot.assign_board(community_card)
            # get win rate
        win_rate = self.bot.estimate_winrate(n=200)
        bet_amount = state.community_state.smallblind
        action = action_table.FOLD
        round_bet = state.community_state.total_preround_pot
        total_bet = state.community_state.totalpot
        big_blind = state.community_state.bigblind
        min_bet = state.community_state.call_price
        player_cnt = state.community_state.num_not_fold_player
        my_chips = state.player_states[playerid].stack
        if min_bet == 0:
            action = action_table.CHECK

        if round_bet == total_bet:
            #nobody bet in this round, default check
            action = action_table.CHECK

        if total_bet < big_blind * 1.5:
            total_bet = big_blind * 1.5

        min_bet_ratio = float(min_bet)/total_bet
        try:
            total_win_rate = win_rate ** (player_cnt-1)
        except ZeroDivisionError:
            total_win_rate = 0
        print("win_rate: {}".format(win_rate))
        print("total_win_rate: {}".format(total_win_rate))

        max_call = 0

        if state.community_state.round == 0: #round_bet=0
            #TODO: consider which round of Deal
            if total_win_rate > 0.8 and win_rate > 0.92:
                print("Deal#1 rule hit, try to attract others to call")
                action = action_table.BET
                bet_amount = random.randint(4,7) * big_blind
                max_call = max(0.5 * my_chips, 20 * big_blind)
            elif total_win_rate > 0.7 and win_rate > 0.8:
                print("Deal#2 rule hit")
                action = action_table.BET
                bet_amount = random.randint(2,5) * big_blind
                max_call = max(0.5 * my_chips, 20 * big_blind)
                print("Deal#2: max_call:{}".format(max_call))
            elif my_chips < 10 * big_blind:
                print("Deal#3 rule hit: be conservative because too poor")
                if min_bet < big_blind and win_rate > 0.3:  #smallBlind case
                    action = action_table.CHECK
            elif win_rate > 0.6:
                print("Deal#4 rule hit")
                action = action_table.BET
                bet_amount = random.randint(1,3) * big_blind
                max_call = random.randint(4,6) * big_blind
            elif win_rate > 0.5:
                print("Deal#5 rule hit")
                action = action_table.CHECK
                max_call = random.randint(2,3) * big_blind
            elif win_rate > 0.3:
                print("Deal#6 rule hit")
                action = action_table.CHECK
                max_call = big_blind
            else:
                print("Deal: no rule hit, fold")

        else:
            if total_win_rate > 0.8 and win_rate > 0.98:
                print("#1 rule hit, try to attract others to call")
                if win_rate == 1:
                    action = action_table.BET
                    bet_amount = my_chips
                elif state.community_state.round == 1:
                    action = action_table.BET
                    bet_amount = 0.6 * round_bet
                elif state.community_state.round == 2:
                    action = action_table.BET
                    bet_amount = round_bet
                elif my_chips > round_bet:  #river
                    action = action_table.BET
                    bet_amount = round_bet
                else:
                    action = action_table.BET
                    bet_amount = my_chips
            elif total_win_rate > 0.7 and win_rate > 0.8:
                print("#2 rule hit")
                action = action_table.BET
                bet_amount = 0.5 * round_bet
                max_call = round_bet
            elif total_win_rate > 0.5:
                print("#4 rule hit")
                if big_blind == min_bet:
                    action = action_table.BET
                    bet_amount = min_bet
                action = action_table.CHECK
                max_call = round_bet * win_rate
            elif win_rate > 0.5:
                print("#6 rule hit")
                action = action_table.CHECK
                max_call = round_bet * 0.5
            elif win_rate > 0.3:
                print("#7 rule hit")
                action = action_table.CHECK
                max_call = big_blind
            else:
                print("no rule hit, fold")

        if bet_amount < min_bet:
            bet_amount = min_bet

        if max_call != 0 and max_call < min_bet:
            print("fold because someone bet too much: min_bet:{} max_call:{}".format(min_bet, max_call))
            action = action_table.FOLD
            bet_amount = 0

        print("decide_action: action:{}, amount:{}, max_call:{}".format(action, bet_amount, max_call))
        return ACTION(action, bet_amount)


    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False