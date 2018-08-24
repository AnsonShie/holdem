import gym
import holdem
import agent
import time
import logging
import numpy as np
import keyboard
import traceback

logger = logging.getLogger('TexasHoldemEnv')
DQN_player_id = 9

def episode(env, n_seats, model_list):
    while True:
        cur_state = env.new_cycle()
        env.render(mode='machine')
        cycle_terminal = False
        try:
            logger.info("reseting all reset state")
            for m in model_list:
                m.reset_state()
        except:
            pass
        
        # (cur_state)
        if env.episode_end:
            break
        action_code = None
        pre_state = None
        initial_stack = cur_state.player_states[DQN_player_id].stack
        while not cycle_terminal:
            # play safe actions, check when no one else has raised, call when raised.
            # print(">>> Debug Information ")
            # print("state(t)")
            # for p in cur_state.player_states:
            #     print(p)
            # print(cur_state.community_state)
            DQN_player_react = False
            cur_pre_act_state = cur_state
            if cur_state.community_state.current_player == DQN_player_id:
                DQN_player_react = True

            actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list)
            cur_state, rews, cycle_terminal, info = env.step(actions)

            if DQN_player_react:
                if action_code is not None and pre_state is not None:
                    if model_list[DQN_player_id].react != 0:
                        model_list[DQN_player_id].remember(pre_state, action_code, 0, cur_pre_act_state, False, DQN_player_id)
                        model_list[DQN_player_id].onlineTrainModel()
                        pre_state = cur_state
                        action_code = model_list[DQN_player_id].react
                    elif model_list[DQN_player_id].react == 0:
                        #model_list[DQN_player_id].remember(pre_state, action_code, cur_state.player_states[DQN_player_id].stack, cur_pre_act_state, False, DQN_player_id)
                        #model_list[DQN_player_id].onlineTrainModel()
                        pre_state = None
                        action_code = None
                else:
                    pre_state = cur_state
                    action_code = model_list[DQN_player_id].react


            # print("action(t), (CALL=1, RAISE=2, FOLD=3 , CHECK=0, [action, amount])")
            # print(actions)

            # print("reward(t+1)")
            # print(rews)
            # print("<<< Debug Information ")
            env.render(mode="machine")
        if action_code is not None and pre_state is not None:
            model_list[DQN_player_id].remember(pre_state, action_code, cur_state.player_states[DQN_player_id].stack - initial_stack, cur_state, True, DQN_player_id)
            model_list[DQN_player_id].onlineTrainModel()
        # print("final state")
        # print(cur_state)

        # total_stack = sum([p.stack for p in env._seats])
        # if total_stack != 10000:
        #     return
    try:
        for p in env.winning_players:
            model_list[p.player_id].estimateReward(p.stack)
    except:
        pass
        
    logger.info("Episode End!!!")
    return np.array([p.stack for p in cur_state.player_states])
    
def new_env():
    env = gym.make('TexasHoldem-v2') # holdem.TexasHoldemEnv(2)
    
    for i in range(10):
        env.add_player(i, stack=3000) # add a player to seat 0 with 1000 "chips"
    return env
    
if __name__ == "__main__":
    env = new_env()
    model_list = list()
    model_list.append(agent.idiotModel()) #0
    model_list.append(agent.mcModel()) #1
    model_list.append(agent.mcModel()) #2
    model_list.append(agent.mcModel()) #3
    model_list.append(agent.mcModel()) #4
    model_list.append(agent.mcModel()) #5
    model_list.append(agent.idiotModel()) #6
    model_list.append(agent.idiotModel()) #7
    model_list.append(agent.idiotModel()) #8
    model_list.append(agent.dqnModel()) #9
    
    try:
        model_list[9].loadModel()
    except:
        pass

    #logger.setLevel(logging.INFO)
    log_iterval = 500
    log_stacks = np.zeros(10)
    stacks = np.zeros(10)
    n_episode = 1
    start_time = time.time()
    
    try:
        while True:
            cur_stacks = episode(env, env.n_seats, model_list)
            log_stacks += cur_stacks
            stacks += cur_stacks
            n_episode += 1
            if n_episode % log_iterval == 0:
                print(log_stacks/log_iterval)
                log_stacks.fill(0.0)
                model_list[9].update_target_model()
            if keyboard.is_pressed('q'):
                print("Interrupt by key. n_episode = {}".format(n_episode))
                break
            env.reset()
    except:
        traceback.print_exc()
    
    etime = (time.time()-start_time)
    print("Elapsed time: {}, per episode {}".format(etime, float(etime)/n_episode))
    print(stacks/n_episode)

    #model_list[9].saveModel()
