#!/usr/bin/env python
# coding: utf-8

# In[1]:


import timeit
import random
import warnings

import numpy as np
import gym
import pytest
import time
import pickle
from nle import nethack, _pynethack


# Load Selected Actions

# In[2]:


ACTIONS = [
    107, #CompassDirection.N 0
    108, #CompassDirection.E 1
    106, #CompassDirection.S 2
    104, #CompassDirection.W 3
    110, #98 CompassDirection.SE 4
    121, #121 CompassDirection.NW 5
    60, #MiscDirection.UP 6
    62, #MiscDirection.DOWN 7
    4 #4 Command.KICK 8
]


# In[3]:


env = gym.make("NetHackScore-v0",actions=ACTIONS)


# In[4]:


def get_views(look,observation):
    
    '''Narrow down observation space'''
    
    x=observation["blstats"][0] 
    y=observation["blstats"][1]
    
    x_look=look
    y_look=look
    
    x_min=x-np.round(x_look/2)
    x_max=x+np.round(x_look/2)

    if x_min < 0:
        x_min=0
        x_max=np.round(x_look/2)*2

    if x_max>(observation["glyphs"].shape[1]):
        x_max=float(observation["glyphs"].shape[1])
        x_min=float(observation["glyphs"].shape[1]-look)


    y_min=y-np.round(y_look/2)
    y_max=y+np.round(y_look/2)

    if y_min < 0:
        y_min=0
        y_max=np.round(y_look/2)*2

    if y_max>(observation["glyphs"].shape[0]):
        y_max=float(observation["glyphs"].shape[0])
        y_min=float(observation["glyphs"].shape[0]-look)
    
    view=observation["glyphs"][int(y_min):int(y_max),int(x_min):int(x_max)] #environment
        
    return view


# # Train for movable spaces and rewards

# In[5]:


dict_movable_space={}
dict_not_movable_space={}
dict_gives_reward={} #only work with positive rewards


# In[6]:


def learn_movable_spaces(observation,dict_movable_space,dict_not_movable_space,dict_gives_reward):
    '''Lern where you can and cant move. Learn what rewards you get'''
    
    #Get current y,x coordinate from view grid, before taking action
    x_ba=observation["blstats"][0] 
    y_ba=observation["blstats"][1]
    
    
    if y_ba==1 or y_ba==20 or x_ba==1 or x_ba==78:
        observation, reward, done, info =env.step(np.random.randint(4))
        
    else:
        view=get_views(10,observation)
        
        #Coordinates of hero
        y_loc_view,x_loc_view=(np.where(view==333))#need change variable
        x_loc_view=x_loc_view[0]
        y_loc_view=y_loc_view[0]

        #UP.RIGHT.DOWN.LEFT, is this correct?
        state_view=[view[y_loc_view-1,x_loc_view],view[y_loc_view,x_loc_view+1],view[y_loc_view+1,x_loc_view],view[y_loc_view,x_loc_view-1]]    

        #take random action, from 4 card. dir. limit to 0 to 4
        action=(np.random.choice(4))

        observation, reward, done, info =env.step(action)
        if done==True:
            #break
            return observation,dict_movable_space,dict_not_movable_space,dict_gives_reward,done

        #coordinates after random action
        x_aa=observation["blstats"][0] 
        y_aa=observation["blstats"][1]

        #Current state(glyphs object)
        moved_state=state_view[action]

        #check if the action you took allowed you to move to a different cell
        if x_aa!=x_ba or y_aa!=y_ba:
            #move_state=observation["glyphs"][y_aa,x_aa]#wont work,just r
            if moved_state not in dict_movable_space.keys():
                dict_movable_space[moved_state]=1
            else:
                dict_movable_space[moved_state]+=1

            # Did you get a reward for moving to a cell? Average out reward. Set > 1 to test.
        if reward>0:
            if moved_state not in dict_gives_reward.keys():
                dict_gives_reward[moved_state]=np.zeros((1,3)) #first col is num times moved, 2nd is sum reward, 3rd is avg reward
                hold_array=dict_gives_reward.get(moved_state,0)
                hold_array[0,0]=1
                hold_array[0,1]=reward
                hold_array[0,2]=hold_array[0,1]/hold_array[0,0]
            else:
                hold_array=dict_gives_reward.get(moved_state,0)
                hold_array[0,0]+=1
                hold_array[0,1]+=reward
                hold_array[0,2]=hold_array[0,1]/hold_array[0,0]

        #If cell remains unchanges    
        if x_aa==x_ba :
            if y_aa==y_ba:
                    if moved_state not in dict_not_movable_space.keys():
                        dict_not_movable_space[moved_state]=1
                    else:
                        dict_not_movable_space[moved_state]+=1

        for key in dict_gives_reward.keys():
            if key in dict_not_movable_space.keys():
                dict_not_movable_space.pop(key)

        for key in dict_movable_space.keys():
            if key in dict_not_movable_space.keys():
                dict_not_movable_space.pop(key)

    return  observation,dict_movable_space,dict_not_movable_space,dict_gives_reward,done
            


# Run training

# In[7]:


for j in range(200):
    env.seed(np.random.randint(31))
    env.reset()
    time.sleep(0.1)
    done=False
    observation, reward, done, info =env.step(0)
    

    for i in range(900):
        observation,dict_movable_space,dict_not_movable_space,dict_gives_reward,done=learn_movable_spaces(observation,dict_movable_space,dict_not_movable_space,dict_gives_reward) 
        if done==True:
            break


# In[8]:


dict_gives_reward.pop(413)
dict_gives_reward.pop(318)


# In[9]:


for key in dict_gives_reward.keys():
    if key in dict_not_movable_space.keys():
        print(key,"screwed")


# In[10]:


for key in dict_movable_space.keys():
    if key in dict_not_movable_space.keys():
        print(key)


# In[11]:


for key in dict_not_movable_space.keys():
    if key in dict_gives_reward.keys():
        print(key)


# Save files

# In[12]:


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[13]:


# save_obj(dict_movable_space,"dict_movable_space_1")
# save_obj(dict_not_movable_space,"dict_not_movable_space_1")
# save_obj(dict_gives_reward,"dict_gives_reward_1")


# # Train for messages

# In[14]:


dict_policy={}


# In[15]:


def random_walk(action,action_opposite):
    while True:   
        if action<4: #up down left right
            next_action=(np.random.choice(4))
            if next_action!=action_opposite[action]:
                break
        else:
            next_action=(np.random.choice(4))
            break
   
    return next_action


# In[16]:


def update_policy(action,reward,key_dict,dict_policy):
    
    '''Update policy network 
    0 is the # of visits
    1 is sum of rewards from vist
    3 is the average reward'''
    
    policy=dict_policy.get(key_dict,0)
    policy[action,0]+=1
    policy[action,1]+=reward
    policy[action,2]=(policy[action,1])/(policy[action,0])
     
    return dict_policy


# In[17]:


def message_policy(key_dict,dict_policy):

    policy=dict_policy.get(key_dict,0)
    
    if np.random.rand(1) <0.25:#Egreedy choice
        next_action=env.action_space.sample()
    else: #arg max choice
        hold=policy[:,2]
        next_action=random.choice(np.array(np.where(hold==np.amax(hold)))[0,:]) 
    return next_action


# In[18]:


for epi in range(100):
    cum_reward=0
    done=False
    count=0
    done=False
    cum_reward=0
    env.seed(np.random.randint(31))
    env.reset();
    action=(np.random.choice(4))#first time action is random
    update_need=False
    action_opposite=[2,3,0,1] #Down Left,Up,Rigth

    while True:
        #action_list.append(action)
        observation, reward, done, info =env.step(action)

        if update_need==True:
            dict_policy=update_policy(action,reward,key_dict,dict_policy)
            update_need=False


        cum_reward+=reward

        #check if policy exisits
        message_string=str(observation["message"])
        key_dict=message_string+str(action)
        if np.sum((observation['message']))!=0: #only store states with messages
            if key_dict not in dict_policy.keys():
                dict_policy[key_dict]=np.zeros((env.action_space.n,3))

        #Walk around untill you encounter a message, i.e. a state in this case
        if np.sum((observation['message']))==0: #Choose any random action, in open space
            next_action=random_walk(action,action_opposite)

        #Message encounted
        if np.sum((observation['message']))!=0: 
            update_need=True
            next_action=message_policy(key_dict,dict_policy)
            action_message=action

        action=next_action

        count+=1
        if count==1000:
            break
        if done==True:
            break


# In[19]:


#save_obj(dict_policy,"dict_policy_1")

