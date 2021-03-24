"""Tests for the DDPG agent."""

from typing import Dict, Sequence
import gym
import acme.wrappers as wrappers
from acme.utils import loggers

from absl.testing import absltest
import acme
from acme import specs
from acme import types
from acme.agents.tf import ddpg
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf


def make_networks(action_spec, policy_layer_sizes=(10, 10), critic_layer_sizes=(10, 10)):

  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  policy_layer_sizes = list(policy_layer_sizes) + [num_dimensions]
  critic_layer_sizes = list(critic_layer_sizes) + [1]

  policy_network = snt.Sequential(
      [networks.LayerNormMLP(policy_layer_sizes), tf.tanh])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes))

  return {
      'policy': policy_network,
      'critic': critic_network,
  }


def test_ddpg():
    # Create a fake environment to test with.
    environment = gym.make('LunarLanderContinuous-v2')
    environment = wrappers.GymWrapper(environment) 
    spec = specs.make_environment_spec(environment)

    agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
    env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

    # Create the networks to optimize (online) and target networks.
    agent_networks = make_networks(spec.actions)

    # Construct the agent.
    agent = ddpg.DDPG(
        environment_spec=spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        logger=agent_logger
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, logger=env_loop_logger)
    loop.run(num_episodes=100)


if __name__ == '__main__':
  test_ddpg()
