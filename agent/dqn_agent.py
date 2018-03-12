from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_to_normal_str
import random 

# from treys import Evaluator, Card, Deck
import numpy as np
import sys
sys.path.append('/Users/championFu//ML_project/treys/treys')
from evaluator import Evaluator
from card import Card
from deck import Deck
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.losses import binary_crossentropy

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

class dqnModel():
    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed":831}

        # total 367 states
        # self._state = [0] * 52 * 2 + [0] * 52 * 5 + [0] *3 # { my 2 card (one hot), community 5 card (one hot), total_pot, my_stack, to_call) ]
        self.state = [0] * 52 + [0] * 2 + [0] * 2  # 56
        # add new initial
        self.action_size = 2
        self.gamma = 0.99    # discount rate
        self.epsilon = 1  # exploration rate
        self.learning_rate = 0.001
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95/1000000
        self.memory = deque(maxlen=10000)
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def update_target_model(self):
        # copy weights from model to target_model
            self.target_model.set_weights(self.model.get_weights())

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    # def update_target_model(self):
    #     # copy weights from model to target_model
    #     self.target_model.load_weights('breakout_pass.h5')

    def _build_model(self):
        model = Sequential()

        model.add(Dense(30, input_dim=len(self.state)))
        model.add(Dense(10, input_dim=30))
        model.add(Dense(self.action_size, input_dim=10))

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
               [total_pot, my_stack, to_call]

    def turn_observation_to_stateJust52_plus2dim(self, observation, playerid):
        card_hot = [0]*52
        my_card = observation.player_states[playerid].hand
        for i in my_card:
            card_hot = self.__turn_card_to_one_hot_returnIndx(i, card_hot)
        community_card = observation.community_card
        for i in community_card:
            card_hot = self.__turn_card_to_one_hot_returnIndx(i, card_hot)
        my_stack = observation.player_states[playerid].stack
        total_pot = observation.community_state.totalpot
        return card_hot + [total_pot, my_stack]


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

    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def evaluateFromState(self, state, playerid):
        # print("state",state.player_states[playerid].hand) 
        evaluator = Evaluator()
        hand = []
        board = []
        # p1_score = evaluator.evaluate(board, player1_hand)
        for i in state.player_states[playerid].hand:
            hand.append(Card.new(card_to_normal_str(i)))
            # print(card_to_normal_str(i))
            # print(hand)

        for j in state.community_card:
            if j != -1:
                # print(card_to_normal_str(j))
                board.append(Card.new(card_to_normal_str(j)))
                # print(board)

        if len(board) == 0:
            rank = evaluator.evaluate(hand, [])
        elif len(board) == 3: 
            rank = evaluator.evaluate(hand, board[:3])
        elif len(board) == 4:
            rank = evaluator.evaluate(hand, board[:4])
        elif len(board) == 5:
            rank = evaluator.evaluate(hand, board[:5])
        rank_class = evaluator.get_rank_class(rank)
        class_string = evaluator.class_to_string(rank_class)
        percentage = 1.0 - evaluator.get_five_card_rank_percentage(rank)  # higher better here
        # print("Player hand = {}, percentage rank among all hands = {}".format(class_string, percentage))
        return [rank,percentage]

    def takeAction(self, state, playerid):
        ''' (Predict/ Policy) Select Action under state'''
        # print("state => ",state)
        # print("playerid => ",playerid)

        rank, percentage = self.evaluateFromState(state, playerid)
        # _stateCards = self.turn_observation_to_stateJust52_plus2dim(state, playerid)
        # _stateCards.append(rank)
        # _stateCards.append(percentage)

        if state.community_card[0] == -1:
            if percentage > 0:
                return ACTION(action_table.RAISE, state.player_states[playerid].stack)
            else:
                if random.random() > 0.3 :
                    return ACTION(action_table.FOLD, 0)
                else:
                    return ACTION(action_table.CALL, state.community_state.to_call)
        else:
            if percentage > 0.8:
                return ACTION(action_table.RAISE, state.player_states[playerid].stack)
            elif percentage > 0.5:
                return ACTION(action_table.RAISE, 50)
            elif percentage > 0.3:
                return ACTION(action_table.CALL, state.community_state.to_call)
            else:
                return ACTION(action_table.FOLD, 0)


    def getReload(self, state):
        '''return `True` if reload is needed under state, otherwise `False`'''
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False

    def sameSuit(self, _stateCards):
        x = np.array(_stateCards[:52])
        print(x)
        _index = np.where(x == 1)
        for i in _index:
            print(i)
        
        input("pause")

    def train(self,memory):
        # len(memory[0]) = 4
        # memory[0][0] state, memory[0][1] action, memory[0][2] reward, memory[0][3] next_state

        # one sample
        # using model predict cur state
        target = self.model.predict(np.array(memory[0][0][0]).reshape(1,56))
        # using target_model predict next state
        t = self.target_model.predict(np.array(memory[0][3][0]).reshape(1,56))

        # target = [[ 109.50458527  605.85266113]]
        # print(memory[0][2])
        # print(memory[0][1][0])
        # print(t)
        # print(np.argmax(t))
        # update cur_action = reward + gamma * best_action of next_state of target_model
        target[0][memory[0][1][0]] = memory[0][2] + self.gamma * (t[0][np.argmax(t)])

        self.model.fit(np.array(memory[0][0][0]).reshape(1,56), target, epochs=1, verbose=0)