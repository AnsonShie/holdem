# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Sam Wenke (samwenke@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import gym
import holdem
import agent

# for memory deque
from collections import deque

def lets_play(env, n_seats, model_list):
  memory = deque(maxlen=10000)
  def model_saveMemory(state, action, reward, next_state):
    memory.append((state, action, reward, next_state))  

  # for dqn_model parameter
  dqnModel_id = 0
  rounds_to_train = 1

  cur_state = env.reset()

  # display the table, cards and all
  env.render(mode='human')

  end_of_game = False
  while not end_of_game:
    cycle_terminal = False
    boolen_NeedToRememberStateT = True
    boolen_NeedToRememberStateT1 = False
    cur_round = env._get_round_number()
    stateT_list = []
    stateT1_list = []
    actionList = []
    begin_money = cur_state.player_states[dqnModel_id].stack
    while not cycle_terminal:
      #  play safe actions, check when no one else has raised, call when raised.
      # actions = holdem.safe_actions(cur_state, n_seats=n_seats)
      
      # print("state(t)")
      # for p in cur_state.player_states:
      #   print(p)
      # print(cur_state.community_state)

      # if dqn_agent do action and also trun into next round, remember state t+1 cur_state
      if cur_round != env._get_round_number():
        cur_round = env._get_round_number()
        # print("Turn into next round:",cur_round)
        if boolen_NeedToRememberStateT1:
          # print("state_t+1:",cur_state)
          boolen_NeedToRememberStateT1 = False
          boolen_NeedToRememberStateT = True

          stateT1_For_neuralNetwork = model_list[dqnModel_id].turn_observation_to_stateJust52_plus2dim(cur_state, dqnModel_id)
          rank, percentage = model_list[dqnModel_id].evaluateFromState(cur_state, dqnModel_id)
          stateT1_For_neuralNetwork.append(rank)
          stateT1_For_neuralNetwork.append(percentage)
          stateT1_list.append(stateT1_For_neuralNetwork)
          # [-3] means stack_t - stack_t+1 > 0
          if stateT_list[-1][-3] - stateT1_list[-1][-3] > 0:
            # play next round
            actionList.append(0)
          else:
            # not play next round
            actionList.append(1)
          # input("pause")


      actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list)

      # if player is dqn_agent, remember state t cur_state
      if cur_state.community_state.current_player == dqnModel_id:
        # print("state_t:",cur_state)
        if boolen_NeedToRememberStateT:
          boolen_NeedToRememberStateT = False
          boolen_NeedToRememberStateT1 = True

          stateT_For_neuralNetwork = model_list[dqnModel_id].turn_observation_to_stateJust52_plus2dim(cur_state, dqnModel_id)
          rank, percentage = model_list[dqnModel_id].evaluateFromState(cur_state, dqnModel_id)
          stateT_For_neuralNetwork.append(rank)
          stateT_For_neuralNetwork.append(percentage)
          stateT_list.append(stateT_For_neuralNetwork)
          # print(stateT_For_neuralNetwork)
          # input("pause")

      # and do next action.
      cur_state, rews, cycle_terminal, info = env.step(actions)

      env.render(mode="machine")

      # if cycle_terminal, remember the difference money
      if cycle_terminal:
        if len(stateT_list) != len(actionList) or len(stateT_list) != len(stateT1_list) or len(actionList) != len(stateT1_list):
          print("Error for state_t action state_t+1 length ")
          break
        else:
          reward = cur_state.player_states[dqnModel_id].stack - begin_money 
          # print("stateT_list:",stateT_list)
          # print("action:",actionList)
          # print("stateT1_list:",stateT1_list)
          # print("reward:",reward)
          model_saveMemory(stateT_list, actionList, reward, stateT1_list)
        # input("pause")
        print("Finish this game")

      if len(memory) > rounds_to_train-1:
        # you can define that how many rounds you want to train your model.
        model_list[dqnModel_id].train(memory)
    # break

env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)

model_list = list()

# start with 4 players
env.add_player(0, stack=3000) # add a player to seat 0 with 1000 "chips"
model_list.append(agent.dqnModel())

env.add_player(1, stack=3000) # add another player to seat 1 with 2000 "chips"
model_list.append(agent.idiotModel())

env.add_player(2, stack=3000) # add another player to seat 2 with 3000 "chips"
model_list.append(agent.idiotModel())

env.add_player(3, stack=3000) # add another player to seat 3 with 1000 "chips"
model_list.append(agent.idiotModel())

# play out a hand
lets_play(env, env.n_seats, model_list)