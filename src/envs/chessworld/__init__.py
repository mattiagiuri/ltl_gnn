from .chessworld import ChessWorld
from .chessworld_8 import ChessWorld8
from .chessworld8_easy import ChessWorld8Easy

from gymnasium.envs.registration import register

register(
    id='ChessWorld-v0',
    entry_point='envs.chessworld.chessworld:ChessWorld',
    kwargs=dict(
        continuous_actions=False
    )
)

register(
    id='ChessWorld-v1',
    entry_point='envs.chessworld.chessworld_8:ChessWorld8',
    kwargs=dict(
        continuous_actions=False
    )
)

register(
    id='ChessWorld-v2',
    entry_point='envs.chessworld.chessworld8_easy:ChessWorld8Easy',
    kwargs=dict(
        continuous_actions=False
    )
)
