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

def lets_play(env, n_seats, model_list):
  # memory = deque(maxlen=10000)
  # def model_saveMemory(state, rews, cycle_terminal, playerid):
  #   if cycle_terminal:
  #     memory.append((state, action, , next_state, done))  
  #   else:

  # for dqn_model parameter
  dqnModel_id = 0
  boolen_NeedToRemember = False

  cur_state = env.reset()

  # display the table, cards and all
  env.render(mode='human')

  end_of_game = False
  while not end_of_game:
    cycle_terminal = False
    cur_round = env._get_round_number()
    print(cur_round)
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
        print("Turn into next round:",cur_round)
        if boolen_NeedToRemember:
          print("state_t+1:",cur_state)
          boolen_NeedToRemember = False
          input("pause")

      actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list)

      # if player is dqn_agent, remember state t cur_state
      if cur_state.community_state.current_player == dqnModel_id:
        print("state_t:",cur_state)
        boolen_NeedToRemember = True

        stateT_For_neuralNetwork = model_list[dqnModel_id].turn_observation_to_stateJust52(cur_state, dqnModel_id)
        rank, percentage = model_list[dqnModel_id].evaluateFromState(cur_state, dqnModel_id)
        stateT_For_neuralNetwork.append(rank)
        stateT_For_neuralNetwork.append(percentage)
        print(stateT_For_neuralNetwork)
        # holdem.model_saveMemory(cur_state, rews, cycle_terminal, action)
        input("pause")

      # and do next action.
      cur_state, rews, cycle_terminal, info = env.step(actions)

      # print("cycle_terminal")
      # print(cycle_terminal)

      # print("info")
      # print(info)

      # print("action(t), (CALL=1, RAISE=2, FOLD=3 , CHECK=0, [action, amount])")
      # print(actions)

      # print("reward(t+1)")
      # print(rews)

      env.render(mode="machine")
      # if cycle_terminal, remember the difference money
      # if cycle_terminal:
        # holdem.model_saveMemory(cur_state, rews, cycle_terminal)
        # print("Finish this game")

    # print("final state")
    # print(cur_state)
    break

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