import numpy as np
import pandas as pd
import ast

import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyModel():
    def __init__(self):
        self.move_model_lr = self.move_model_layer1()
        self.move_model_ud = self.move_model_layer2()
        self.attack_model = self.att_model()

        #self.att_model = self.att_model()
        self.possible_actions = pd.read_csv('actions_cleaned.csv')
        self.punches = ['X', 'Y', 'Z']
        self.kicks = ['A', 'B', 'C']
    
    def move_model_layer1(self):
        #Antecedent range
        dir_range = np.arange(-210, 211, 1) #direction
        reach_range = np.arange(19, 211, 1) #reach
        #Consequent range
        move_range = np.arange(0, 1.1, 0.1)

        #Antecedent(s)
        dir = ctrl.Antecedent(dir_range, 'dir')
        reach = ctrl.Antecedent(reach_range, 'reach')

        #direction membership function
        dir['left'] = fuzz.trimf(dir_range, [-210, -210, 0])
        dir['right'] = fuzz.trimf(dir_range, [0, 210, 210])
        #reach membership function
        reach['close'] = fuzz.trapmf(reach_range, [19, 19, 30, 50])
        reach['far'] = fuzz.trapmf(reach_range, [25, 75, 210, 210])
        #Consequent(s)
        move = ctrl.Consequent(move_range, 'move')

        #position mebership function
        move['left'] = fuzz.trimf(move_range, [0, 0, 0.4])
        move['none'] = fuzz.trapmf(move_range, [0.2, 0.4, 0.6, 0.8])
        move['right'] = fuzz.trimf(move_range, [0.6, 1, 1])

        #define rules
        rule1 = ctrl.Rule(dir['left'] & reach['close'], move['none'])
        rule2 = ctrl.Rule(dir['left'] & reach['far'], move['right'])
        rule3 = ctrl.Rule(dir['right'] & reach['close'], move['none'])
        rule4 = ctrl.Rule(dir['right'] & reach['far'], move['left'])

        move_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        move_sim = ctrl.ControlSystemSimulation(move_ctrl)

        return move_sim
    
    def move_model_layer2(self):
        #Antecedent range
        lr_range = np.arange(0, 1.1, 0.1)      #left/right
        state_range = np.arange(512, 524, 2)    #enemy state
        #Consequent range
        move_range = np.arange(0, 1.1, 0.1)

        #Antecedent(s)
        lr = ctrl.Antecedent(lr_range, 'lr')
        state = ctrl.Antecedent(state_range, 'state')

        #lr membership function
        lr['left'] = fuzz.trimf(lr_range, [0, 0, 0.4])
        lr['none'] = fuzz.trapmf(lr_range, [0.2, 0.4, 0.6, 0.8])
        lr['right'] = fuzz.trimf(lr_range, [0.6, 1, 1])
        #state membership function
        state['default'] = fuzz.trapmf(state_range, [512, 512, 514, 516])
        state['jump'] = fuzz.trapmf(state_range, [514, 514, 518, 520])
        state['att'] = fuzz.trapmf(state_range, [516, 518, 522, 522])

        #Consequent(s)
        move = ctrl.Consequent(move_range, 'move')

        #move mebership function
        move['left'] = fuzz.trimf(move_range, [0, 0, 0.3])
        move['none'] = fuzz.trimf(move_range, [0.2, 0.3, 0.4])
        move['up'] = fuzz.trapmf(move_range, [0.3, 0.4, 0.5, 0.6])
        move['down'] = fuzz.trapmf(move_range, [0.5, 0.6, 0.7, 0.8])
        move['right'] = fuzz.trimf(move_range, [0.7, 1, 1])

        #define rules
        rule1 = ctrl.Rule(lr['left'] & state['default'], move['right'])
        rule2 = ctrl.Rule(lr['left'] & state['jump'], move['right'])
        rule3 = ctrl.Rule(lr['left'] & state['att'], move['left'])

        rule4 = ctrl.Rule(lr['none'] & state['default'], move['none'])
        rule5 = ctrl.Rule(lr['none'] & state['jump'], move['up'])
        rule6 = ctrl.Rule(lr['none'] & state['att'], move['down'])

        rule7 = ctrl.Rule(lr['right'] & state['default'], move['left'])
        rule8 = ctrl.Rule(lr['right'] & state['jump'], move['left'])
        rule9 = ctrl.Rule(lr['right'] & state['att'], move['right'])

        move_ctrl = ctrl.ControlSystem(
            [rule1, rule2, rule3, rule4, rule5,
             rule6, rule7, rule8, rule9]
        )
        move_sim = ctrl.ControlSystemSimulation(move_ctrl)

        return move_sim
        
    def att_model(self):
        #Antecedent range
        move_range = np.arange(0, 1.1, 0.1)     #move
        reach_range = np.arange(19, 211, 1)     #reach
        #Consequent range
        attack_range = np.arange(0, 1.1, 0.05)

        #Antecedent(s)
        move = ctrl.Antecedent(move_range, 'move')
        reach = ctrl.Antecedent(reach_range, 'reach')

        #move memebrship function
        move['left'] = fuzz.trimf(move_range, [0, 0, 0.3])
        move['none'] = fuzz.trimf(move_range, [0.2, 0.3, 0.4])
        move['up'] = fuzz.trapmf(move_range, [0.3, 0.4, 0.5, 0.6])
        move['down'] = fuzz.trapmf(move_range, [0.5, 0.6, 0.7, 0.8])
        move['right'] = fuzz.trimf(move_range, [0.7, 1, 1])
        #reach membership function
        reach['close'] = fuzz.trapmf(reach_range, [19, 19, 30, 50])
        reach['far'] = fuzz.trapmf(reach_range, [25, 75, 210, 210])
        
        #Consequent(s)
        attack = ctrl.Consequent(attack_range, 'attack')

        #attack mebership function
        attack['punch'] = fuzz.trapmf(attack_range, [0, 0, 0.3, 0.4])
        attack['none'] = fuzz.trimf(attack_range, [0.3, 0.5, 0.7]) 
        attack['kick'] = fuzz.trapmf(attack_range, [0.6, 0.7, 1, 1])

        #define rules
        rule1 = ctrl.Rule(move['left'] & reach['far'], attack['none'])
        rule2 = ctrl.Rule(move['left'] & reach['close'], attack['punch'])

        rule3 = ctrl.Rule(move['right'] & reach['far'], attack['none'])
        rule4 = ctrl.Rule(move['right'] & reach['close'], attack['punch'])

        rule5 = ctrl.Rule(move['down'] & reach['far'], attack['kick'])
        rule6 = ctrl.Rule(move['down'] & reach['close'], attack['kick'])

        rule7 = ctrl.Rule(move['up'] & reach['far'], attack['punch'])
        rule8= ctrl.Rule(move['up'] & reach['close'], attack['punch'])

        rule9 = ctrl.Rule(move['none'], attack['punch'])

        att_ctlr = ctrl.ControlSystem(
            [rule1, rule2, rule3, rule4, 
             rule5, rule6, rule7, rule8, rule9]
        )
        att_sim = ctrl.ControlSystemSimulation(att_ctlr)

        return att_sim

    def compute_move(self, agent_x, enemy_x, enemy_state):
        layer1_res = self.compute_move_layer1(agent_x, enemy_x)

        layer2_res = self.comput_move_layer2(layer1_res, enemy_state)

        if layer2_res < 0.2:
            move_action = 'RIGHT'
        elif layer2_res < 0.4:
            move_action = 'IDLE'
        elif layer2_res < 0.6:
            move_action = 'UP'
        elif layer2_res < 0.8:
            move_action = 'DOWN'
        else:
            move_action = 'LEFT'

        return move_action, layer2_res
    
    def compute_move_layer1(self, agent_x, enemy_x):
        self.move_model_lr.input['dir'] = agent_x - enemy_x
        self.move_model_lr.input['reach'] = abs(agent_x - enemy_x)
        
        self.move_model_lr.compute()

        result = self.move_model_lr.output['move']

        return result
    
    def comput_move_layer2(self, lr_res, enemy_state):
        self.move_model_ud.input['lr'] = lr_res
        self.move_model_ud.input['state'] = enemy_state
        
        self.move_model_ud.compute()

        result = self.move_model_ud.output['move']
        
        return result

    def compute_att(self, move_res, agent_x, enemy_x):
        self.attack_model.input['move'] = move_res
        self.attack_model.input['reach'] = abs(agent_x - enemy_x)    

        self.attack_model.compute()

        result = self.attack_model.output['attack']

        if result < 0.4:
            attack_action = self.punches[np.random.randint(len(self.punches))]
        elif result < 0.65:
            attack_action = 'IDLE'
        else:
            attack_action = self.kicks[np.random.randint(len(self.kicks))]
        
        return attack_action
    
    def compute_action(self, move_action, att_action):
        df = self.possible_actions

        action_index = df.index[(df['move_defs'] == move_action) & (df['att_defs'] == att_action)]
        action_data = ast.literal_eval(df.iloc[action_index[0], 0])

        return action_data