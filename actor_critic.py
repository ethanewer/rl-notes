from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.types import Device
from tqdm import trange  # type: ignore


def plot_rewards(rewards: dict[str, list[float]]) -> None:
    for label, r in rewards.items():
        average_rewards = np.convolve(r, np.ones(50) / 50, mode="full")[: len(r)]
        plt.plot(average_rewards, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()


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
        running = [[True] * batch_size]
        actions = []
        rewards = []
        with torch.no_grad():
            while any(running[-1]):
                logits = policy(states[-1])
                probs = F.softmax(logits, dim=-1)
                actions.append(torch.multinomial(probs, num_samples=1)[:, 0])

                states.append(torch.zeros_like(states[-1]))
                rewards.append(
                    torch.zeros(batch_size, dtype=torch.float32, device=device)
                )
                running.append([])
                for i in range(batch_size):
                    if running[-2][i]:
                        next_state, reward, term, trunc, _ = envs[i].step(
                            actions[-1][i].item()
                        )
                        states[-1][i] = torch.from_numpy(next_state).to(device)
                        rewards[-1][i] = float(reward)
                        running[-1].append(not (term or trunc))
                    else:
                        running[-1].append(False)

        s = torch.stack(states[:-1])
        a = torch.stack(actions)
        r = torch.stack(rewards)

        mask = torch.tensor(running[:-1], device=device)

        policy_opt.zero_grad()

        logits = policy(s)
        policy_loss = torch.tensor(0.0)
        for i in range(batch_size):
            policy_loss += loss_fn(
                logits[mask[:, i], i], a[mask[:, i], i], r[mask[:, i], i]
            )

        policy_loss.backward()
        policy_opt.step()
        reward_records.append(r.sum(dim=1).mean().item())

    for env in envs:
        env.close()

    return reward_records


def optimize_actor_critic(
    device: Device = "cpu",
    gamma: float = 0.99,
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

    value = nn.Sequential(
        nn.Linear(n_observations, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    ).to(device)

    policy_opt = optim.AdamW(policy.parameters(), lr=1e-4)  # type: ignore
    value_opt = optim.AdamW(value.parameters(), lr=1e-4)  # type: ignore

    reward_records: list[float] = []

    for _ in trange(n_iters, desc="actor critic optimization", leave=False):
        reward_sum = 0.0
        states = torch.from_numpy(np.array([env.reset()[0] for env in envs])).to(device)
        running = [True] * batch_size
        while any(running):
            with torch.no_grad():
                logits = policy(states)
                probs = F.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, num_samples=1)[:, 0]

            next_states = torch.zeros_like(states)
            rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
            for i in range(batch_size):
                if running[i]:
                    next_state, reward, term, trunc, _ = envs[i].step(actions[i].item())
                    next_states[i] = torch.from_numpy(next_state).to(device)
                    rewards[i] = float(reward)
                    running[i] = not (term or trunc)

            a = actions[running]
            s = states[running]
            next_s = next_states[running]
            r = rewards[running]

            value_opt.zero_grad()
            value_loss = F.mse_loss(
                value(s)[:, 0], r + gamma * value(next_s)[:, 0].detach()
            )
            value_loss.backward()
            value_opt.step()

            policy_opt.zero_grad()
            with torch.no_grad():
                advantage = r + gamma * value(next_s)[:, 0] - value(s)[:, 0]

            policy_loss = torch.mean(
                F.cross_entropy(policy(s), a, reduction="none") * advantage
            )

            policy_loss.backward()
            policy_opt.step()

            states = next_states
            if any(running):
                reward_sum += float(np.mean(rewards[running].numpy()))

        reward_records.append(reward_sum)

    for env in envs:
        env.close()

    return reward_records
