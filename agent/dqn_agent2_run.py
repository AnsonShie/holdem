from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import random

from treys import Evaluator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.losses import binary_crossentropy
from collections import deque
import os
import logging
from agent.mcbot.mcbot import MonteCarloBot

logging.basicConfig(level=logging.DEBUG)

CHAR_SUIT_TO_INT = {
        's': 0,  # spades
        'h': 13,  # hearts
        'd': 26,  # diamonds
        'c': 39,  # clubs
}
# " {'23456789TJQKA'} + {'shdc''} (note: lower case) "
CHAR_NUM_TO_INT = {
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4,
        '6': 5,
        '7': 6,
        '8': 7,
        '9': 8,
        'T': 9,
        'J': 10,
        'Q': 11,
        'K': 12,
        'A': 0,
}

class Action():
    def __init__(self, state):
        self.state = state

    def Fold(self):
        if self.state.community_state.total_preround_pot == self.state.community_state.totalpot:
            print('CHECK')
            return 2, ACTION(action_table.CHECK, 0)
        else:
            print('FOLD')
            return 0, ACTION(action_table.FOLD, 0)

    def Bet(self, round_bet, big_blind, min_bet, my_chips, round_id, win_rate):
        if round_id == 0:
            bet_amount = random.randint(2, 7) * big_blind
            max_call = max(0.5 * my_chips, 20 * big_blind)
        elif round_id == 1:
            max_call = win_rate * round_bet
            bet_amount = win_rate * 0.6 * round_bet
        else:
            max_call = win_rate * round_bet
            bet_amount = win_rate * round_bet

        if win_rate <= 0.5:
            return self.Fold()

        if bet_amount < min_bet:
            bet_amount = min_bet
        if max_call < min_bet:
            return self.Fold()
        else:
            print("BET")
            return 1, ACTION(action_table.BET, bet_amount)

    def Check(self, round_bet, big_blind, min_bet, round_id, win_rate):
        if round_id == 0:
            max_call = random.randint(1, 3) * big_blind
        else:
            max_call = win_rate * round_bet
        if max_call < min_bet:
            return self.Fold()
        else:
            print("CHECK")
            return 2, ACTION(action_table.CHECK, min_bet)

class dqnModel_run():
    bot = MonteCarloBot()
    # https://keon.io/deep-q-learning/

    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}

        # total 367 states
        # { my 2 card (one hot), community 5 card (one hot), player_cnt, total_pot, to_call, win_rate, (River, Turn, Flop, Deal) ]
        self._state = [0] * 52 * 2 + [0] * 52 * 5 + [0] * 4 + [0] * 4
        # add new initial
        self.action_size = 3
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.ModelDir = 'Model_run/'
        self.ModelName = 'DQNmodel.h5'

        self.win_rate = 0
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.08  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.react = 0
        #self.update_target_model()
    
    def get_ModelPath(self):
        if not os.path.isdir(self.ModelDir):
            os.mkdir(self.ModelDir)
        return self.ModelDir + self.ModelName
        

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
    def remember(self, state, action, reward, next_state, done, playerid):
        state = self.__turn_observation_to_state(state, playerid)
        next_state = self.__turn_observation_to_state(next_state, playerid)
        
        state = np.array(state).reshape(1,len(self._state))
        next_state = np.array(next_state).reshape(1,len(self._state))
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, playerid):
        state = self.__turn_observation_to_state(state, playerid)
        state = np.array(state).reshape(1,len(self._state))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def _replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.load_weights(self.get_ModelPath())

    def _build_model(self):
        model = Sequential()

        model.add(Dense(100, input_dim=len(self._state)))
#         model.add(Dense(100, input_shape=(len(self._state),)))
        model.add(Dense(50, input_dim=100))
        model.add(Dense(self.action_size, input_dim=50))

        opt = Adam(lr=self.learning_rate)
        #opt=RMSprop(lr=self.learning_rate,decay=0.99)
        model.compile(loss=self._huber_loss, optimizer=opt)
        model.summary()
        return model

    def __turn_card_to_one_hot(self, card):
        if card == -1:
            return [0] * 52
        # " {'23456789TJQKA'} + {'shdc''} (note: lower case) "
        card_info = card_to_normal_str(card)
        
        # for j in ['s','h','d','c']:
        #     for i in ['A','2','3','4','5','6','7','8','9','T','J','Q','K']:
        #         card_hot = [0]*52
        #         print('{0}'.format(i+j))
        #         card_idx = CHAR_NUM_TO_INT[i] + CHAR_SUIT_TO_INT[j]
        #         print(card_idx)
        #         card_hot[card_idx] = 1
        #         print(card_hot)
        #         input('pause')

        card_idx = CHAR_NUM_TO_INT[card_info[:1]] + CHAR_SUIT_TO_INT[card_info[1:]]
        card_hot = [0]*52
        card_hot[card_idx] = 1
        return card_hot

    def __turn_observation_to_state(self, observation, playerid):
        my_card = observation.player_states[playerid].hand
        community_card = observation.community_card
        round_num = observation.community_state.round
        round_one_hot = [0, 0, 0, 0]
        if round_num == 0:
            round_one_hot = [0, 0, 0, 1]
        elif round_num == 1:
            round_one_hot = [0, 0, 1, 0]
        elif round_num == 2:
            round_one_hot = [0, 1, 0, 0]
        elif round_num == 3:
            round_one_hot = [1, 0, 0, 0]

        my_card_list = [card_to_normal_str(card) for card in my_card]
        community_card_list = [card_to_normal_str(card) for card in community_card if card != -1]
        self.bot.reset()
        self.bot.assign_hand(my_card_list)
        if len(community_card_list) > 0:
            # get board cards
            self.bot.assign_board(community_card_list)
            # get win rate
        self.win_rate = self.bot.estimate_winrate(n=200)

        player_cnt = observation.community_state.num_not_fold_player
        my_stack = observation.player_states[playerid].stack
        total_pot = observation.community_state.totalpot
        to_call = observation.community_state.to_call
        return self.__turn_card_to_one_hot(my_card[0]) + \
               self.__turn_card_to_one_hot(my_card[1])+ \
               self.__turn_card_to_one_hot(community_card[0])+ \
               self.__turn_card_to_one_hot(community_card[1])+ \
               self.__turn_card_to_one_hot(community_card[2])+ \
               self.__turn_card_to_one_hot(community_card[3])+ \
               self.__turn_card_to_one_hot(community_card[4])+ \
               [player_cnt, total_pot, to_call, self.win_rate] + round_one_hot

    def __turn_observation_to_stateJust52(self, observation, playerid):
        card_hot = [0]*52
        my_card = observation.player_states[playerid].hand
        for i in my_card:
            card_hot = self.__turn_card_to_one_hot_returnIndx(i, card_hot)
        community_card = observation.community_card
        for i in community_card:
            card_hot = self.__turn_card_to_one_hot_returnIndx(i, card_hot)
        my_stack = observation.player_states[playerid].stack
        total_pot = observation.community_state.totalpot
        to_call = observation.community_state.to_call
        return card_hot + [total_pot, my_stack, to_call]


    def __turn_card_to_one_hot_returnIndx(self, card, card_hot):
        if card == -1:
            return card_hot
        else:
            card_info = card_to_normal_str(card)
            card_idx = CHAR_NUM_TO_INT[card_info[:1]] + CHAR_SUIT_TO_INT[card_info[1:]]
            if card_hot[card_idx] == 1:
                input("Error!!!!!! card_hot cann't duplicate")
            else:
                card_hot[card_idx] = 1
            return card_hot

    def batchTrainModel(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def onlineTrainModel(self):
        state, action, reward, next_state, done = self.memory[-1]
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if len(self.memory) % 20 == 0:
            self.saveModel()
        

    def saveModel(self):
        self.model.save_weights(self.get_ModelPath())

    def loadModel(self):
        if os.path.isfile(self.get_ModelPath()):
            self.model.load_weights(self.get_ModelPath())

    def takeAction(self, state, playerid):
        ''' (Predict/ Policy) Select Action under state'''
        # print("state => ",state)
        # print("playerid => ",playerid)

        # input node: 55
        #_stateCards = self.__turn_observation_to_stateJust52(state, playerid)
        #print("Test State => ", _stateCards)
        # input("pause")
        
        # input node: 367 (52*2 + 52*5 + 3)
        _stateCards = self.__turn_observation_to_state(state, playerid)
#         print("Test State => ", self.__turn_observation_to_state(state, playerid))

        #self.remember(state, action, reward, state, done, playerid)

        action = Action(state)
        round_bet = state.community_state.total_preround_pot
        big_blind = state.community_state.bigblind
        min_bet = state.community_state.call_price
        my_chips = state.player_states[playerid].stack
        round_id = state.community_state.round

        self.react = self.act(state, playerid)
        if self.react == 0:
            self.react, return_info = action.Fold()
            return return_info
        elif self.react == 1:
            self.react, return_info = action.Bet(round_bet, big_blind, min_bet, my_chips, round_id, self.win_rate)
            return return_info
        else:
            self.react, return_info = action.Check(round_bet, big_blind, min_bet, round_id, self.win_rate)
            return return_info
        

    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False

    def sameSuit(self, _stateCards):
        x = np.array(_stateCards[:53])
        print(x)
        print(np.where(x == 1))
        
        input("pause")
