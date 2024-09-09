import retro
from gym import Env 
from gym.spaces import MultiBinary, Box

import numpy as np
import math
import cv2

# Create custom environment 
class StreetFighter(Env): 
    def __init__(self):
        super().__init__()
        # Specify action space and observation space 
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        # Startup and instance of the game 
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        
        # Starting health in Street Fighter II is 176 HP
        self.START_HEALTH = 176
        self.REWARD_COEFF = 17
        self.PENALTY_COEFF = 1.75
        self.LOSS_PENALTY_COEFF = 0.35

        self.enemy_wins = 0
        self.player_wins = 0
        self.pos_reward = 0
        self.neg_reward = 0
        
        # enemy and player health values that get updated as the game goes along
        self.enemy_health = self.START_HEALTH
        self.player_health = self.START_HEALTH

        # Creating a score variable to hold the player's current score; important for calculating the reward on each step
        self.score = 0
        
    def reset(self):
        # Return the first frame 
        obs = self.game.reset()
        obs = self.preprocess(obs) 
        self.previous_frame = obs 
        
        self.enemy_wins = 0
        self.player_wins = 0
        self.pos_reward = 0
        self.neg_reward = 0
        # Create a attribute to hold the score delta 
        self.score = 0
        self.enemy_health = self.START_HEALTH
        self.player_health = self.START_HEALTH
        
        return obs
    
    def preprocess(self, observation): 
        # Grayscaling 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize 
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84,84,1))
        return channels 
    
    def step(self, action): 
        # Take a step 
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs) 
        
        # Frame delta 
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs 
        
        # Reshape the reward function
         # calculating the change in health for each player (i.e. health from this frame - health from previous frame)
        enemy_damage_taken = abs(info['enemy_health'] - self.enemy_health)
        player_damage_taken = abs(info['health'] - self.player_health)
        
        # Tweaking the reward function to be the score of this step - score from the previous step (i.e. the change in score)

        # catching edge cases to make sure no reward is being earned outside of a fight (i.e. in between rounds)
        if (self.enemy_health != 0 and info['enemy_health'] == 0 and self.player_health != 0 and info['health'] == 0) or (enemy_damage_taken == 0 and player_damage_taken == 0) or (self.player_health == 0 and self.enemy_health == 0):
            reward = 0
        
        # If the player wins and enemy loses
        elif info['enemy_health'] < 0:
            self.player_wins += 1
            reward = self.START_HEALTH * math.log(info['health'], self.START_HEALTH) * self.REWARD_COEFF
            
        # if the enemy wins and player loses
        elif info['health'] < 0:
            self.enemy_wins += 1
            reward = -math.pow(self.START_HEALTH, (info['enemy_health'] / self.START_HEALTH)) * self.LOSS_PENALTY_COEFF
                               
        # the fight goes on
        else:
            # If the enemy took more damage than the player
            if enemy_damage_taken > player_damage_taken:
                reward = ((enemy_damage_taken) - (player_damage_taken)) * self.REWARD_COEFF
            # If the player took more or same amount of damage than the enemy
            else:
                reward = ((enemy_damage_taken) - (player_damage_taken)) * self.PENALTY_COEFF

        #update current health
        self.enemy_health = info['enemy_health']
        self.player_health = info['health']
        
        #update current score to compare with next state
        self.score = info['score']

        if reward < 0:
            self.neg_reward += 1
        elif reward > 0:
            self.pos_reward += 1
        
        return frame_delta, reward, done, info
    
    def render(self, *args, **kwargs):
        self.game.render()
        
    def close(self):
        self.game.close()