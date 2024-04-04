import numpy as np
from dm_control import suite, viewer

import envs.dmc as dmc

dmc.register_with_suite()


def main():
    env = suite.load(domain_name="ltl_cartpole", task_name="ltl", visualize_reward=True)
    action_spec = env.action_spec()

    def random_policy(time_step):
        print(time_step.observation['propositions'])
        print(time_step.observation['terminated'])
        return np.random.uniform(low=action_spec.minimum,
                                 high=action_spec.maximum,
                                 size=action_spec.shape)

    viewer.launch(env, policy=random_policy)


if __name__ == '__main__':
    main()
