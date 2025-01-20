from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces

import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium.core import ActType, ObsType

from ltl.logic import Assignment


class ChessWorld(gym.Env):
    PIECES = {
        "queen": (4, 0),
        "rook": (4, 2),
        "knight": (1, 3),
        "bishop": (3, 4),
        "pawn": (2, 2)
    }

    ATTACKED_SQUARES = {
        "queen": {(4, 0), (4, 1), (4, 2), (3, 1), (2, 2),
                  (3, 0), (2, 0), (1, 0), (0, 0)},
        "rook": {(4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
                 (3, 2), (2, 2)},
        "knight": {(1, 3), (3, 2), (2, 1), (0, 1), (3, 4)},
        "bishop": {(0, 1), (1, 2), (2, 3), (3, 4), (4, 3)},
        "pawn": {(2, 2), (3, 1), (3, 3)}

    }

    FREE_SQUARES = {(0, 2), (1, 1), (0, 3), (0, 4), (1, 4), (2, 4)}

    def __init__(self, continuous_actions=False):
        # print((set([(i, j) for i in range(5) for j in range(5)]) -
        #              set.union(*[attacked for attacked in self.ATTACKED_SQUARES.values()])
        #             ))
        assert (self.FREE_SQUARES ==
                    (set([(i, j) for i in range(5) for j in range(5)]) -
                     set.union(*[attacked for attacked in self.ATTACKED_SQUARES.values()])
                    )
                )
        self.rng = np.random.default_rng()
        self.continuous_actions = continuous_actions
        self.delta_t = 0.08

        self.observation_space = spaces.MultiDiscrete([5, 5])
        if continuous_actions:
            self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(9)

        self.agent_pos = np.array([0, 4])  # will be updated in reset

    def get_active_propositions(self) -> set[str]:
        props = set()
        pos = tuple(self.agent_pos)

        for piece in self.PIECES.keys():
            if pos in self.ATTACKED_SQUARES[piece]:
                props.add(piece)

        return props

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if options and 'init_square' in options:
            self.agent_pos = np.array(options['init_square'])
            return self.agent_pos.copy(), {'propositions': set()}

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = np.array(self.rng.choice(list(self.FREE_SQUARES)))
        return self.agent_pos.copy(), {'propositions': set()}

    def initial_square_reset(self) -> tuple[ObsType, dict[str, Any]]:
        self.agent_pos = np.array((0, 4))
        return self.agent_pos.copy(), {'propositions': set()}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        if not self.continuous_actions:
            if action == 0:
                action = np.array([0, 1])
            elif action == 1:
                action = np.array([1, 0])
            elif action == 2:
                action = np.array([0, -1])
            elif action == 3:
                action = np.array([-1, 0])
            elif action == 4:
                action = np.array([1, 1])
            elif action == 5:
                action = np.array([1, -1])
            elif action == 6:
                action = np.array([-1, 1])
            elif action == 7:
                action = np.array([-1, -1])
            elif action == 8:
                action = np.array([0, 0])
            else:
                raise ValueError(f"Invalid action: {action}")

        self.agent_pos = self.agent_pos + action
        terminated = False
        reward = 0.0
        if (self.agent_pos < 0).any() or (self.agent_pos > 4).any():
            terminated = True
            reward = -1.0
        return self.agent_pos.copy(), reward, terminated, False, {'propositions': self.get_active_propositions()}

    def get_propositions(self):
        return sorted([piece for piece in self.PIECES])

    def get_possible_assignments(self) -> list[Assignment]:
        props = set(self.get_propositions())
        return [
            Assignment.where('queen', propositions=props),
            Assignment.where('rook', propositions=props),
            Assignment.where('knight', propositions=props),
            Assignment.where('bishop', propositions=props),
            Assignment.where('pawn', propositions=props),
            Assignment.where('queen', 'rook', propositions=props),
            Assignment.where('queen', 'pawn', propositions=props),
            Assignment.where('queen', 'rook', 'pawn', propositions=props),
            # Assignment.where('rook', 'pawn', propositions=props),
            Assignment.where('rook', 'knight', propositions=props),
            Assignment.where('rook', 'bishop', propositions=props),
            Assignment.where('knight', 'bishop', propositions=props),
            Assignment.zero_propositions(props),
        ]

    def squares_of_assignments(self, reach_or_avoid_set):
        squares_list = []

        for active_props in reach_or_avoid_set:
            intersection = set.intersection(*[self.ATTACKED_SQUARES[prop] for prop in active_props])
            others = set.union(*[self.ATTACKED_SQUARES[prop] for prop in self.PIECES.keys()
                                 if prop not in active_props])

            squares_list.append(intersection - others)

        return set.union(*squares_list)

    # @staticmethod
    # def render(trajectory: list[np.ndarray] = None, ax=None):
    #     if trajectory is None:
    #         trajectory = []
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1)
    #     for circle in FlatWorld.CIRCLES:
    #         xy = (float(circle.center[0]), float(circle.center[1]))
    #         patch = plt.Circle(xy, circle.radius, color=circle.color, fill=True, alpha=.2)
    #         ax.add_patch(patch)
    #
    #     if len(trajectory) > 0:
    #         trajectory = np.array(trajectory)
    #         ax.plot(trajectory[:, 0], trajectory[:, 1], color='green', marker='o',
    #                 linestyle='dashed',
    #                 linewidth=2, markersize=1)
    #         ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], s=100, marker='o', c="orange")
    #         ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], s=100, marker='o', c="g")
    #     ax.axis('square')
    #     hide_ticks(ax.xaxis)
    #     hide_ticks(ax.yaxis)
    #     ax.set_xlim([-2.1, 2.1])
    #     ax.set_ylim([-2.1, 2.1])

    @staticmethod
    def render(trajectory: list[np.ndarray] = None, ax=None):
        pass


def hide_ticks(axis):
    for tick in axis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)


if __name__ == '__main__':
    env = ChessWorld(continuous_actions=False)
    obs, _ = env.reset()
    trajectory = [obs]
    for i in range(300):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        trajectory.append(obs)
        if term or trunc:
            break
    env.render(trajectory)
    # plt.show()

