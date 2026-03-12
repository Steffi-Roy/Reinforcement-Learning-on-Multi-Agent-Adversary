
from pettingzoo.mpe import simple_adversary_v3
import pygame

env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=False, dynamic_rescaling=False,render_mode="human")
agent = env.agents[adversary_0, agent_1, agent_0]
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy to select an action for each agent


    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
