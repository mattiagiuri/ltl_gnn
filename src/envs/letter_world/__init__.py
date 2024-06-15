from gymnasium.envs.registration import register

register(
    id='LetterEnv-v0',
    entry_point='envs.letter_world.letter_env:LetterEnv',
    kwargs=dict(
        grid_size=7,
        letters="aabbccddeeffgghhiijjkkll",
        use_fixed_map=False,
        use_agent_centric_view=True,
    )
)

register(
    id='LetterEnvNonMyopic-v0',
    entry_point='envs.letter_world.letter_env:LetterEnv',
    kwargs=dict(
        grid_size=7,
        letters="abcdefghijkl",
        use_fixed_map=True,
        use_agent_centric_view=True,
        map={
            (3, 0): 'a',
            (0, 2): 'a',
            (4, 0): 'b'
        }
    )
)


register(
    id='LetterEnvTry-v0',
    entry_point='envs.letter_world.letter_env:LetterEnv',
    kwargs=dict(
        grid_size=7,
        letters="abcdefghijkl",
        use_fixed_map=True,
        use_agent_centric_view=True,
        map={
            (1, 0): 'a',
            (0, 2): 'b',
        }
    )
)
