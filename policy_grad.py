from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.types import Device
from tqdm import trange  # type: ignore


@torch.no_grad()
def sample_action(state: NDArray, policy: nn.Module, device: Device) -> float:
    inputs = torch.tensor(state[None], dtype=torch.float).to(device)
    logits = policy(inputs)[0]
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_trajectory(
    env: gym.Env,
    policy: nn.Module,
    device: Device,
) -> tuple[list[NDArray], list[int], list[float]]:
    done = False
    states: list[NDArray] = []
    actions: list[int] = []
    rewards: list[float] = []
    s, _ = env.reset()
    while not done:
        states.append(s.tolist())
        a = sample_action(s, policy, device)
        s, r, term, trunc, _ = env.step(a)
        actions.append(int(a))
        rewards.append(float(r))
        done = term or trunc

    return states, actions, rewards


def optimize_policy_gradient(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: Device = "cpu",
    n_iters: int = 512,
) -> list[float]:
    env = gym.make("CartPole-v1")

    n_actions = env.action_space.n  # type: ignore
    state, _ = env.reset()
    n_observations = len(state)

    policy = nn.Sequential(
        nn.Linear(n_observations, 64),
        nn.GELU(),
        nn.Linear(64, n_actions),
    ).to(device)

    opt = optim.AdamW(policy.parameters(), lr=0.001)  # type: ignore

    reward_records: list[float] = []

    for _ in trange(n_iters, desc="Optimizing Policy"):
        states, actions, rewards = sample_trajectory(env, policy, device)

        s = torch.tensor(states).to(device)
        a = torch.tensor(actions).to(device)
        r = torch.tensor(rewards).to(device)

        opt.zero_grad()
        loss = loss_fn(policy(s), a, r)
        loss.backward()
        opt.step()

        reward_records.append(sum(rewards))

    env.close()
    return reward_records


def plot_rewards(rewards: dict[str, list[float]]) -> None:
    for label, r in rewards.items():
        average_rewards = np.convolve(r, np.ones(50) / 50, mode="full")[: len(r)]
        plt.plot(average_rewards, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
