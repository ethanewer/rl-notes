import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.types import Device
from tqdm import trange  # type: ignore


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

    policy_opt = optim.AdamW(policy.parameters(), lr=1e-5)  # type: ignore
    value_opt = optim.AdamW(value.parameters(), lr=1e-5)  # type: ignore

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

            a = actions
            s = states
            next_s = next_states
            r = rewards

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
                reward_sum += float(np.mean(rewards.numpy()))

        reward_records.append(reward_sum)

    for env in envs:
        env.close()

    return reward_records
