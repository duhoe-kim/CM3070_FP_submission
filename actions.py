#file to extract action space from gym
import retro

import ast
import pandas as pd

move_defs = ['RIGHT', 'LEFT', 'UP', 'DOWN']
att_defs = ['A', 'B', 'C', 'Z', 'X', 'Y']

def convert_int_action(int_num):
    bin_num = '{0:012b}'.format(int_num)

    action = [int(d) for d in bin_num]

    return action

def extract_actions(env, filename):
    df_action = pd.DataFrame(columns=['actions', 'definitions'])

    for i in range(0, 4096):
        action = convert_int_action(i)

        action_def = env.unwrapped.get_action_meaning(action)

        if len(action_def) != 0:
            df_action.loc[len(df_action.index)] = [action, action_def]

    filepath = filename + '.csv'
    df_action.to_csv(filepath, index=False)

def preprocess_data(filename):
    filepath = filename + '.csv'
    df = pd.read_csv(filepath)

    df.drop_duplicates(subset=['definitions'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    new_df = pd.DataFrame(columns=['actions', 'move_defs', 'att_defs'])
    
    del_count = 0
    
    action = []
    move_def = ''
    att_def = ''

    for i in range(len(df.index)):
        action = df.iloc[i - del_count, 0]
        defs = df.iloc[i - del_count, 1]

        res = ast.literal_eval(defs)

        if len(res) > 2:
            df = df.drop(index = i)
            del_count += 1

        elif len(res) == 1:
            found = False
            for j in range(len(move_defs)):
                if res[0] == move_defs[j] and found == False:
                    move_def = res[0]
                    att_def = 'IDLE'
                    found = True
                elif j == len(move_defs) - 1 and found == False:
                    move_def = 'IDLE'
                    att_def = res[0]

            new_df.loc[len(new_df.index)] = [action, move_def, att_def]

        elif len(res[0]) == len(res[1]):
            df = df.drop(index = i)
            del_count += 1

        else:
            found = False
            for j in range(len(move_defs)):
                if res[0] == move_defs[j] and found == False:
                    move_def = res[0]
                    att_def = res[1]
                    found = True
                elif j == len(move_defs) - 1 and found == False:
                    move_def = res[1]
                    att_def = res[0]

            if len(att_def) > 1:
                df = df.drop(index = i)
                del_count += 1
            else:
                new_df.loc[len(new_df.index)] = [action, move_def, att_def]

    new_df = new_df.sort_values(by='move_defs')
    new_df.to_csv('actions_cleaned.csv', index=False)

def save_actions_csv():
    env = retro.make(
        game = 'StreetFighterIISpecialChampionEdition-Genesis'
    )
    filename = 'actions'

    extract_actions(env, filename)
    preprocess_data(filename)

save_actions_csv()
