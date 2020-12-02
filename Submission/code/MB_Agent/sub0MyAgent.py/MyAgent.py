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
import pathlib
path = str(pathlib.Path(__file__).parent.absolute())

# In[2]:


# Load Dicts.

# In[49]:


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[50]:

action_list=[1]
dict_movable_space=np.load(path+'/dict_movable_space.npy',allow_pickle='TRUE').item()
dict_not_movable_space=np.load(path+'/dict_not_movable_space.npy',allow_pickle='TRUE').item()
dict_gives_reward=np.load(path+'/dict_gives_reward.npy',allow_pickle='TRUE').item()
dict_policy=np.load(path+'/dict_policy.npy',allow_pickle='TRUE').item()

# In[6]:


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


# In[47]:


# env = gym.make("NetHackScore-v0",actions=ACTIONS)


# In[48]:


def get_views(look,observation):
    
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





# # Train for messages

# In[51]:


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


# In[52]:


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


# In[53]:


def message_policy(key_dict,dict_policy):
    policy=dict_policy.get(key_dict,0)
    if np.random.rand(1) <0.25:#Egreedy choice
        next_action=env.action_space.sample()
    else: #arg max choice
        hold=policy[:,2]
        next_action=random.choice(np.array(np.where(hold==np.amax(hold)))[0,:]) 
    return next_action


# # Training

# In[113]:


def get_action_reward(observation):
    
    x_ba=observation["blstats"][0] 
    y_ba=observation["blstats"][1]
    if y_ba<=1 or y_ba>=19 or x_ba<=1 or x_ba>=77:
#         print("bounds, random")
        next_action=np.random.randint(4)
    
    else:
        view=get_views(10,observation)
#         print("view")
#         print(view)

        reward_list=[]
        for rr in np.unique(view):
            if rr in dict_gives_reward.keys():
                reward_list.append(rr)
        loc_reward=np.array(np.where(view==reward_list[0]))
        y_loc_view,x_loc_view=(np.where(view==333))
        
        state_view=[[y_loc_view-1,x_loc_view],[y_loc_view,x_loc_view+1],[y_loc_view+1,x_loc_view],[y_loc_view,x_loc_view-1]]
        state_view_char=[view[y_loc_view-1,x_loc_view],view[y_loc_view,x_loc_view+1],view[y_loc_view+1,x_loc_view],view[y_loc_view,x_loc_view-1]]   

        goal_distance=[]
        for i in range(len(state_view)):
            a=(state_view[i])
            b=loc_reward
            goal_distance.append(np.linalg.norm(a-b))

        action_array=np.array([0,1,2,3])
        action_sort= list(map(lambda x, y,z:(x,y,z), goal_distance, action_array,state_view_char))
        action_sort.sort(key=lambda tup: tup[0], reverse=False)
        
        

        hold_list=[]
        for j in range(len(action_sort)):
            possible=action_sort[j][2][0]
            action_take=action_sort[j][1]
            
#             print("possible:",possible)

            if possible in dict_not_movable_space.keys():
#                 print("cant move:",possible)
                1+1
            else:
                hold_list.append(action_take)

        if np.random.rand(1) <0.25:#Egreedy choice
#             print("random action")
            hold_list[0]=np.random.choice(4)#env.action_space.sample()
        
        next_action=hold_list[0]
#         print("return",next_action)

    return next_action


# In[114]:


def get_message_action(observation,action):
    message_string=str(observation["message"])
    key_dict=message_string+str(action)
    if key_dict not in dict_policy.keys():
        dict_policy[key_dict]=np.zeros((env.action_space.n,3))
        
    policy=dict_policy.get(key_dict,0)
    if np.random.rand(1) <0.25:#Egreedy choice
        next_action=np.random.randint(9)
    else:
        hold=policy[:,2]
        next_action=random.choice(np.array(np.where(hold==np.amax(hold)))[0,:]) 
 
    return next_action


# In[115]:


def walk(observation,visited_observation):

    x_ba=observation["blstats"][0] 
    y_ba=observation["blstats"][1]
    if y_ba<=1 or y_ba>=19 or x_ba<=1 or x_ba>=77:
        next_action=np.random.randint(4)
    
    else:
        global_observation=observation["glyphs"]
        next_action=100

        #get neighbour
        y,x=(np.where(global_observation==333))#need change variable
        y=y[0]
        x=x[0]
        
#         print("loc of guy",y,x)
        
        #UP.RIGHT.DOWN.LEFT
        neighbour_loc=[(y-1,x),(y,x+1),(y+1,x),(y,x-1)]
        neighbour_visited_state=[visited_observation[y-1,x],visited_observation[y,x+1],visited_observation[y+1,x],visited_observation[y,x-1]]
        
        if 0 in neighbour_visited_state:
#             print("zero found")
            #UP
            if (neighbour_visited_state[0]==0) and global_observation[neighbour_loc[0]] not in dict_not_movable_space.keys():
                next_action=0
#                 print("taking up")
            #Right
            elif neighbour_visited_state[1]==0 and global_observation[neighbour_loc[1]] not in dict_not_movable_space.keys():
                    next_action=1
#                     print("taking right")
            #down
            elif neighbour_visited_state[2]==0 and global_observation[neighbour_loc[2]] not in dict_not_movable_space.keys():
                    next_action=2
#                     print("taking down")
            #left
            elif neighbour_visited_state[3]==0 and global_observation[neighbour_loc[3]] not in dict_not_movable_space.keys():
                    next_action=3
#                     print("taking left")
            else:
                next_action=np.random.randint(4)#limit to 4
#                 print("random 4 null")
        else:
            next_action=np.random.randint(4)#limit to 4

    return next_action

action_used_num=[0,1,2,3,4,5,6,7,8]
action_map=[1,2,3,4,6,8,17,18,20]
# In[155]:


# In[10]:


class MyAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.seeds = kwargs.get('seeds', None)
        self.visited_observation=np.zeros((21,79))

    def act(self, observation):
        
        
        x_cur=observation["blstats"][0] 
        y_cur=observation["blstats"][1]
        self.visited_observation[y_cur,x_cur]=1
        

        if any(item in np.unique(get_views(10,observation)) for item in dict_gives_reward.keys()): #reward
#             print("found reward")
            next_action=get_action_reward(observation)
        elif np.sum((observation['message']))!=0: #message
#             print("message")
            next_action=np.random.randint(4)#get_message_action(observation,action_list[-1])
        else:#walk
#             print("walk")
            next_action=walk(observation,self.visited_observation)

        action=next_action
        
#         print("action",action)
        
        action_list.append(action)

        return action_map[action_used_num.index(action)]


# In[14]:
