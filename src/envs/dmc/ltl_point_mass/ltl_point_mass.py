# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

import collections

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.rl.control import _spec_from_observation

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    with open(f'envs/dmc/ltl_point_mass/point_mass.xml', mode='rb') as f:
        xml_string = f.read()
    return xml_string, common.ASSETS


@SUITE.add('ltl')
def ltl(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the LTL task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = LTLPointMass(randomize_gains=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics for the point_mass domain."""

    def point_mass_position(self):
        return self.named.data.geom_xpos['pointmass']


class LTLPointMass(base.Task):
    """The PointMass LTL task"""

    def __init__(self, randomize_gains=False, random=None):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._randomize_gains = randomize_gains
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

           If _randomize_gains is True, the relationship between the controls and
           the joints is randomized, so that each control actuates a random linear
           combination of joints.

        Args:
          physics: An instance of `mujoco.Physics`.
        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        if self._randomize_gains:
            dir1 = self.random.randn(2)
            dir1 /= np.linalg.norm(dir1)
            # Find another actuation direction that is not 'too parallel' to dir1.
            parallel = True
            while parallel:
                dir2 = self.random.randn(2)
                dir2 /= np.linalg.norm(dir2)
                parallel = abs(np.dot(dir1, dir2)) > 0.9
            physics.model.wrap_prm[[0, 1]] = dir1
            physics.model.wrap_prm[[2, 3]] = dir2
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        obs['propositions'] = self.get_propositions(physics)
        obs['terminated'] = False
        return obs

    def get_reward(self, physics):
        return 0

    def observation_spec(self, physics):
        # This is a hack, since the label and terminated information is not actually part of the observation
        observation = self.get_observation(physics)
        del observation['propositions']
        del observation['terminated']
        return _spec_from_observation(observation)

    @staticmethod
    def get_propositions(physics):
        pos = physics.point_mass_position()
        x, y = pos[0], pos[1]
        if .05 <= x <= .25 and .05 <= y <= .25:
            return ['green']
        elif -.25 <= x <= -.05 and .05 <= y <= .25:
            return ['blue']
        elif -.25 <= x <= -.05 and -.25 <= y <= -.05:
            return ['red']
        elif .05 <= x <= .25 and -.25 <= y <= -.05:
            return ['yellow']
        return []
