# NetHack Project
NetHack Wiki: (Useful info.)<br>
https://nethackwiki.com/

NLE API: <br>
https://github.com/facebookresearch/nle 

NLE Paper: <br>
https://arxiv.org/pdf/2006.13760.pdf


Nethack example Agent: <br>
https://github.com/facebookresearch/nle/blob/master/nle/agent/agent.py

## NLE default options :
- Human male neutral monk

- NETHACKOPTIONS = [
    "color",
    "showexp",
    "autopickup",
    "pickup_types:$?!/",
    "pickup_burden:unencumbered",
    "nobones",
    "nolegacy",
    "nocmdassist",
    "disclose:+i +a +v +g +c +o",
    "runmode:teleport",
    "mention_walls",
    "nosparkle",
    "showexp",
    "showscore",
] <br>
This means we don't have to pickup items.

## The action space:
The action space might be extended to included all 90'something actions, 
By default, the action space includes the following that might be stepped in gym:
- [0] : More (Not doing anything)
- [1] : North 1 step <br>
- [2] : East 1 step <br>
- [3] : South 1 step <br>
- [4] : West 1 step <br>
- [5] : North-East 1 step <br>
- [6] : Sout-East 1 step <br>
- [7] : South-West 1 step <br>
- [8] : North-West 1 step <br>
- [9] : North max <br>
- [10] : East max <br>
- [11] : South max <br>
- [12] : West max <br>
- [13] : North-East max <br>
- [14] : Sout-East max <br>
- [15] : South-West max <br>
- [16] : North-West max <br>
- [17] : Go up a staircase <br>
- [18] : Go down a starcase <br>
- [19] : Wait / Do nothing <br>
- [20] : Kick <br>
- [21] : Eat <br>
- [22] : Search <br>

To reduce the action space, I'm removing the actions that auto move (0, and 9 - 16)


## The observation space (We have to make sure the blstats expanded here are correct.):
If we work with the same setup as the example agent we have the following: <br>
observation_space['glyphs'] = Box(0, 5976, (21, 79), int16), which may represent a symbol with int val between 0 and 5976 in the shape (height=21, width=79)  <br>

observation_space['blstats'] = Box(-something, +something, (25, ), int16), which are 25 stats in an array: <br>
- [0] : X_Coordinate
- [1] : Y_Coordinate <br>
- [2] : Strength Percentage<br>
- [3] : Strength (Strength corresponds to the ability to have more weight in your inventory.) <br>
- [4] : Dexterity (has a multitude of effects, of which the most significant is probably that it affects your chance of hitting monsters, whether in melee combat or with a missile or spell) <br>
- [5] : Constitution (Having a high constitution increases your healing rate and the number of HP you gain when levelling up and allows you to carry more weight in your inventory.)<br>
- [6] : Intelligence (If you are a Healer, Knight, Monk, Priest or Valkyrie, in which case it is wisdom that affects your chances of successfully casting a spe<br>
- [7] : Wisdom  (A Healer, Knight, Monk, Priest or Valkyrie requires wisdom to cast spells) <br>
- [8] : Charisma (Charisma is mostly useful for obtaining better prices at shops. )<br>
- [9] : Score <br>
- [10] : Current Health Points <br>
- [11] : Maximum Health Points <br>
- [12] : Dungeon depth  <br>
- [13] : Available gold <br>
- [14] : Current energy <br>
- [15] : Max energy <br>
- [16] : Armor class <br>
- [17] : Monster level <br>
- [18] : Experience level <br>
- [19] : Experience points <br>
- [20] : Time <br>
- [21] : Hunger level (Too little and you starve; too much and you choke.) <br>
- [22] : Carying capacity <br>
- [23] : NLE stat  <br>
- [24] : NLE stat   <br>

To avoid overfitting, I am only trying x & y coords, Hp devided by MaxXp, and Hunger level  (Considered score, but if you think about it, the model should not learn based on the current scores.)

#### Observations are not normalized
Due to the symbollic representation, the observations should not be normalized. 

## Rewards
Maybe its a good idea to clip the rewards as the NLE paper did using tanh(r/100). They also note that intrinsic rewards, such as gold value etc do not yield good results,
and it better to use extrinsic rewards. (From Gym reward feedback.) The rewards are computed as new_score - old_score


### First tryout seems promising
********************************************************
steps: 668
episodes: 2
mean 100 episode reward: -5.0

********************************************************
 
********************************************************
steps: 853
episodes: 3
mean 100 episode reward: -1.2

********************************************************
 
********************************************************
steps: 1369
episodes: 4
mean 100 episode reward: -1.9

********************************************************
 
********************************************************
steps: 1939
episodes: 5
mean 100 episode reward: -2.5

********************************************************
 
********************************************************
steps: 2314
episodes: 6
mean 100 episode reward: -2.5

********************************************************
 
********************************************************
steps: 3251
episodes: 7
mean 100 episode reward: -3.2

********************************************************
 
********************************************************
steps: 4064
episodes: 8
mean 100 episode reward: -3.1

********************************************************
 
********************************************************
steps: 4219
episodes: 9
mean 100 episode reward: -2.3

********************************************************
 
********************************************************
steps: 5490
episodes: 10
mean 100 episode reward: -0.6

********************************************************
 
********************************************************
steps: 7894
episodes: 11
mean 100 episode reward: 6.5

********************************************************
 
********************************************************
steps: 8525
episodes: 12
mean 100 episode reward: 6.0

********************************************************

Process finished with exit code 137





