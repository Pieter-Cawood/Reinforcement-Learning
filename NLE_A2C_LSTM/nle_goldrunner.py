import enum

import numpy as np
from nle.env import base
from nle import nethack


TASK_ACTIONS = tuple(
    [nethack.MiscAction.MORE]
    + list(nethack.CompassDirection)
    + list(nethack.CompassDirectionLonger)
    + list(nethack.MiscDirection)
    + [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
)


class NetHackScore(base.NLE):
    """Environment for "score" task.
    The task is an augmentation of the standard NLE task. The return function is
    defined as:
    :math:`\text{score}_t - \text{score}_{t-1} + \text{TP}`,
    where the :math:`\text{TP}` is a time penalty that grows with the amount of
    environment steps that do not change the state (such as navigating menus).
    Args:
        penalty_mode (str): name of the mode for calculating the time step
            penalty. Can be ``constant``, ``exp``, ``square``, ``linear``, or
            ``always``. Defaults to ``constant``.
        penalty_step (float): constant applied to amount of frozen steps.
            Defaults to -0.01.
        penalty_time (float): constant applied to amount of frozen steps.
            Defaults to -0.0.
    """

    def __init__(
            self,
            *args,
            penalty_mode="constant",
            penalty_step: float = -0.01,
            penalty_time: float = -0.0,
            **kwargs,
    ):
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time

        self._frozen_steps = 0

        actions = kwargs.pop("actions", TASK_ACTIONS)
        super().__init__(*args, actions=actions, **kwargs)

    def _get_time_penalty(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[20]  # moves
        new_time = blstats_new[20]  # moves

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0

        penalty = 0
        if self.penalty_mode == "constant":
            if self._frozen_steps > 0:
                penalty += self.penalty_step
        elif self.penalty_mode == "exp":
            penalty += 2 ** self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "square":
            penalty += self._frozen_steps ** 2 * self.penalty_step
        elif self.penalty_mode == "linear":
            penalty += self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "always":
            penalty += self.penalty_step
        else:  # default
            raise ValueError("Unknown penalty_mode '%s'" % self.penalty_mode)
        penalty += (new_time - old_time) * self.penalty_time
        return penalty

    def _reward_fn(self, last_observation, observation, end_status):
        """Score delta, but with added a state loop penalty."""
        score_diff = super()._reward_fn(last_observation, observation, end_status)
        time_penalty = self._get_time_penalty(last_observation, observation)
        return score_diff + time_penalty


class NetHackGoldRunner(NetHackScore):
    """Environment for the "goldrunner" task.
    The task is similar to the one defined by `NetHackScore`, but the score is
    defined by the changes in glyphs discovered by the agent and the amount of gold collected
    """

    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        return super().reset(*args, **kwargs)

    def _reward_fn(self, last_observation, observation, end_status):
        del end_status  # Unused

        if not self.env.in_normal_game():
            # Before game started or after it ended stats are zero.
            return 0.0

        reward = 0
        glyphs = observation[self._glyph_index]
        old_blstats = last_observation[self._blstats_index]
        blstats = observation[self._blstats_index]

        dungeon_num, dungeon_level = blstats[23:25]

        old_gold = old_blstats[13]
        gold = blstats[13]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != 0)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        time_penalty = self._get_time_penalty(last_observation, observation)
        return reward + time_penalty + (gold - old_gold) * 3
