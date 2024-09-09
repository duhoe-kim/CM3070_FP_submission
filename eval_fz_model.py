import retro

import numpy as np
import pandas as pd
import keyboard
import time

from env_wrapper import StreetFighter
from fz_model import FuzzyModel

def run_step():
    env = retro.make(
        game = 'StreetFighterIISpecialChampionEdition-Genesis'
    )
    model = FuzzyModel()

    obs = env.reset()

    action_input = env.action_space.sample()
    obs, reward, done, info = env.step(action_input)

    move_action, move_res = model.compute_move(
        info['agent_x'], info['enemy_x'], info['enemy_status']
    )

    att_action = model.compute_att(
        move_res, info['agent_x'], info['enemy_x'], info['enemy_status']
    )
    
    print(move_action, att_action)

    action = model.compute_action(move_action, att_action)
    
    print(action)


def run_sim():
    env = StreetFighter()
    model = FuzzyModel()

    df_record = pd.DataFrame(columns=['rewards', 'actions'])
    df_compare = pd.DataFrame(columns=['pos_rewards', 'neg_rewards', 'player_wins', 'enemy_wins'])

    # Reset game to starting state
    env.reset()
    # Set flag to flase
    done = False
    
    action_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for game in range(1): 
        #while game status is not finished
        while not done:
            #escape using key 'q'
            if keyboard.is_pressed('q'):
                df_record.to_csv('fz_record.csv', index=False)
                
                df_compare.loc[len(df_compare.index)] = [env.pos_reward, env.neg_reward, env.player_wins, env.enemy_wins]
                df_compare.to_csv('fz_compare.csv', index=False)
                
                env.close()
                break
            #escape after game is over
            if done:
                df_record.to_csv('fz_record.csv', index=False)
                
                df_compare.loc[len(df_compare.index)] = [env.pos_reward, env.neg_reward, env.player_wins, env.enemy_wins]
                df_compare.to_csv('fz_compare.csv', index=False)

                obs = env.reset()
            
            #render environment with most recent state
            env.render()
            #return feedback in the step
            obs, reward, done, info = env.step(action_input)
            time.sleep(0.01)

            try:
                move_action, move_res = model.compute_move(
                    info['agent_x'], info['enemy_x'], info['enemy_status']
                )

                att_action = model.compute_att(
                    move_res, info['agent_x'], info['enemy_x']
                )

                action_input = model.compute_action(move_action, att_action)
                
                df_record.loc[len(df_record.index)] = [reward, action_input] 
            except Exception as e:
                #if error => no action
                action_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            print(env.pos_reward, env.neg_reward, env.player_wins, env.enemy_wins)
run_sim()



    