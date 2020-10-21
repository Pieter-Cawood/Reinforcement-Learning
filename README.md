# NetHack Project
API https://github.com/facebookresearch/nle <br>
Paper https://arxiv.org/pdf/2006.13760.pdf


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


## The observation space (We have to make sure the blstats are correct.):
If we work with the same setup as the example agent we have the following: <br>
observation_space['glyphs'] = Box(0, 5976, (21, 79), int16), which may represent a symbol with int val between 0 and 5976 in the shape (height=21, width=79)  <br>

observation_space['blstats'] = Box(-something, +something, (25, ), int16), which are 25 stats in an array: <br>
- [0] : X_Coordinate
- [1] : Y_Coordinate (Strength) <br>
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
- [25] : NLE stat  <br>

To avoid overfitting, I am only trying x & y coords, Hp devided by MaxXp, and Hunger level  (Considered score, but if you think about it, the model should not learn based on the current scores.)

#### Observations are not normalized
Due to the symbollic representation, the observations should not be normalized. 

## Rewards
Maybe its a good idea to clip the rewards as the NLE paper did using tanh(r/100). They also note that intrinsic rewards, such as gold value etc do not yield good results,
and it better to use extrinsic rewards. (From Gym reward feedback.)





