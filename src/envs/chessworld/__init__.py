from .chessworld import ChessWorld

from gymnasium.envs.registration import register

register(
    id='ChessWorld-v0',
    entry_point='envs.chessworld.chessworld:ChessWorld',
    kwargs=dict(
        continuous_actions=False
    )
)