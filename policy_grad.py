from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.types import Device
from tqdm import trange  # type: ignore


def optimize_policy_gradient(
    loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    device: Device = "cpu",
    n_iters: int = 256,
    batch_size: int = 32,
) -> list[float]:
    envs = [gym.make("CartPole-v1") for _ in range(batch_size)]

    n_actions = envs[0].action_space.n  # type: ignore
    state, _ = envs[0].reset()
    n_observations = len(state)

    policy = nn.Sequential(
        nn.Linear(n_observations, 64),
        nn.GELU(),
        nn.Linear(64, n_actions),
    ).to(device)

    policy_opt = optim.AdamW(policy.parameters(), lr=1e-3)  # type: ignore

    reward_records: list[float] = []

    for _ in trange(n_iters, desc="policy gradient optimization", leave=False):
        states = [
            torch.from_numpy(np.array([env.reset()[0] for env in envs])).to(device)
        ]
        running = [True] * batch_size
        actions = []
        rewards = []
        with torch.no_grad():
            while any(running):
                logits = policy(states[-1])
                probs = F.softmax(logits, dim=-1)
                actions.append(torch.multinomial(probs, num_samples=1)[:, 0])

                states.append(torch.zeros_like(states[-1]))
                rewards.append(
                    torch.zeros(batch_size, dtype=torch.float32, device=device)
                )
                for i in range(batch_size):
                    if running[i]:
                        next_state, reward, term, trunc, _ = envs[i].step(
                            actions[-1][i].item()
                        )
                        states[-1][i] = torch.from_numpy(next_state).to(device)
                        rewards[-1][i] = float(reward)
                        running[i] = not (term or trunc)

        s = torch.stack(states[:-1])
        a = torch.stack(actions)
        r = torch.stack(rewards)

        policy_opt.zero_grad()

        logits = policy(s)
        policy_loss = torch.tensor(0.0)
        for i in range(batch_size):
            policy_loss += loss_fn(logits[:, i], a[:, i], r[:, i])

        policy_loss.backward()
        policy_opt.step()
        reward_records.append(r.sum(dim=0).mean().item())

    for env in envs:
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
