import ppo
import torch
import numpy as np
from gym.spaces import Box, Discrete

def test_make_env():
    for env_name in ['CartPole-v1', 'LunarLander-v2']:
        env = ppo.make_env(env_name)
        obs = env.reset()
        env.close()
    print("Test make_env: SUCCESSFUL")


def test_mlp():
    IN = 1
    HIDDEN = (10, 10)
    OUT = 5
    BATCH = 10

    sizes = [IN] + list(HIDDEN) + [OUT]
    mlp = ppo.mlp(sizes)
    inp = torch.ones((BATCH, IN), dtype=torch.float32)
    out = mlp(inp)

    if out.shape == torch.Size((BATCH, OUT)):
        print("Test mlp: SUCCESSFUL")
    else:
        print("Test mlp: FAILED")


def test_MLPCategoricalActor():
    OBS_DIM = 8
    ACT_DIM = 2
    HIDDEN_SIZES = (100, 100)
    BATCH = 10

    actor = ppo.MLPCategoricalActor(OBS_DIM, ACT_DIM, HIDDEN_SIZES)

    obs_batch = torch.zeros((BATCH, OBS_DIM), dtype=torch.float32)
    act_batch = torch.zeros(BATCH)

    pi, logp_a = actor(obs_batch, act=act_batch)

    if pi.logits.shape == torch.Size((BATCH, ACT_DIM)) and logp_a.shape == torch.Size((BATCH,)):
        print("Test MLPCategoricalActor: SUCCESSFUL")
    else:
        print("Test MLPCategoricalActor: FAILED")

def test_MLPCritic():
    OBS_DIM = 8
    HIDDEN_SIZES = (100, 100)
    BATCH = 10

    critic = ppo.MLPCritic(OBS_DIM, HIDDEN_SIZES)

    obs_batch = torch.zeros((BATCH, OBS_DIM), dtype=torch.float32)

    values = critic(obs_batch)

    if values.shape == torch.Size((BATCH,)):
        print("Test MLPCritic: SUCCESSFUL")
    else:
        print("Test MLPCritic: FAILED")


def test_MLPCategoricalActorCritic():
    ENV = ppo.make_env("CartPole-v1")
    OBS_SPACE = ENV.observation_space
    OBS_DIM = OBS_SPACE.shape[0]
    ACT_SPACE = ENV.action_space
    HIDDEN_SIZES = (100, 100)
    BATCH = 10

    ac = ppo.MLPActorCritic(OBS_SPACE, ACT_SPACE, HIDDEN_SIZES)

    obs_batch = torch.zeros((BATCH, OBS_DIM), dtype=torch.float32)
    act_batch = torch.zeros(BATCH)

    a_batch, v_batch, logp_a_batch = ac.step(obs_batch)

    obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    act = ac.act(obs)

    if not (a_batch.shape == (BATCH,) and v_batch.shape == (BATCH,) and logp_a_batch.shape == (BATCH,)):
        print("Test MLPActorCritic (Categorical): FAILED")
    elif not (act.shape == (1,)):
        print("Test MLPActorCritic (Categorical): FAILED")
    else:
        print("Test MLPActorCritic (Categorical): SUCCESSFUL")

def test_cumsum():
    buffer = ppo.Buffer(10, 2, 10)
    x = np.ones(5)
    discount = 1
    y = buffer._discount_cumsum(x, 1)
    if np.sum(y - np.array([5, 4, 3, 2, 1], dtype=np.float32)) == 0:
        print("Test CumSum: SUCCESSFUL")
    else:
        print("Test CumSum: FAILED")

def test_Buffer():
    ENV = ppo.make_env("CartPole-v1")
    OBS_DIM = ENV.observation_space.shape
    if isinstance(ENV.action_space, Box):
        ACT_DIM = ENV.action_space.shape[0]
    elif isinstance(ENV.action_space, Discrete):
        ACT_DIM = ENV.action_space.n   
    SIZE = 1000
    STEPS = 1000
    MAX_EP_LEN = 50

    buffer = ppo.Buffer(OBS_DIM, ACT_DIM, SIZE)

    obs = ENV.reset()
    ep_len = 0
    for i in range(STEPS):
        o, r, d, _ = ENV.step(ENV.action_space.sample())
        v, logp, a = 1, 2, 3
        buffer.store(o, a, r, v, logp)

        timeout = ep_len == MAX_EP_LEN
        terminal = d or timeout
        epoch_ended = i == SIZE

        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

            # if trajectory didn't reach terminal state, bootstrap value target
            if timeout or epoch_ended:
                _, v, _ = 1, 2, 3

            else:
                v = 0

            buffer.finish_path(v)

            o, ep_len = ENV.reset(), 0

    if not buffer.ptr == 1000:
        print("Test Buffer: FAILED")
        return
    data = buffer.get()
    if data['obs'].shape != torch.Size((SIZE, *OBS_DIM)):
        print("Test Buffer: FAILED")
        return
    print("Test Buffer: SUCCESSFUL")


def test_PPOAgent():
    ENV = ppo.make_env("CartPole-v1")

    agent = ppo.PPOAgent(ENV)

    print("Test PPOAgent: SUCCESSFUL")


if __name__ == "__main__":
    test_make_env()
    test_mlp()
    test_MLPCategoricalActor()
    test_MLPCritic()
    test_MLPCategoricalActorCritic()
    test_cumsum()
    test_Buffer()
    test_PPOAgent()