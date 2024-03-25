from dataclasses import dataclass

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
                 env_net: nn.Module,
                 ltl_net: nn.Module,
                 actor: nn.Module,
                 critic: nn.Module
                 ):
        super().__init__()
        self.env_net = env_net
        self.ltl_net = ltl_net
        self.actor = actor
        self.critic = critic
        self.recurrent = False

    def forward(self, obs):
        env_embedding = self.env_net(obs.features)
        ltl_embedding = self.ltl_net(obs.ltl)
        embedding = torch.cat([env_embedding, ltl_embedding], dim=1)

        dist = self.actor(embedding)
        value = self.critic(embedding).squeeze(1)
        return dist, value
