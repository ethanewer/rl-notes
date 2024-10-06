from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore
from flax import nnx
from numpy.typing import NDArray
from tqdm import trange  # type: ignore

import jax
from jax import Array, nn
from jax import numpy as jnp


class MLP(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs) -> None:
        self.fc_in = nnx.Linear(in_features, 64, rngs=rngs)
        self.fc_out = nnx.Linear(64, out_features, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.fc_out(nnx.gelu(self.fc_in(x)))


def sample_action(state: NDArray, policy: MLP, key: Array) -> Array:
    logits = policy(jnp.array(state[None]))[0]
    probs = nn.softmax(logits, axis=-1)
    return jax.random.choice(key, a=jnp.arange(len(probs)), p=probs)


def sample_trajectory(
    env: gym.Env,
    policy: MLP,
    key: Array,
) -> tuple[list[list[int]], list[int], list[float], Array]:
    done = False
    states: list[list[int]] = []
    actions: list[int] = []
    rewards: list[float] = []
    s, _ = env.reset()
    while not done:
        states.append(s.tolist())
        key, subkey = jax.random.split(key)
        a = int(sample_action(s, policy, subkey))
        s, r, term, trunc, _ = env.step(a)
        actions.append(a)
        rewards.append(float(r))
        done = term or trunc

    return states, actions, rewards, key


def optimize_policy_gradient(
    loss_fn: Callable[[MLP, Array, Array, Array], Array],
    n_iters: int,
) -> list[float]:
    env = gym.make("CartPole-v1")

    n_actions = env.action_space.n  # type: ignore
    state, _ = env.reset()
    n_observations = len(state)

    policy = MLP(n_observations, n_actions, rngs=nnx.Rngs(0))

    opt = nnx.Optimizer(policy, optax.adamw(1e-3))

    reward_records: list[float] = []

    key = jax.random.PRNGKey(0)

    # @nnx.jit
    def train_step(
        policy: MLP,
        opt: nnx.Optimizer,
        s: Array,
        a: Array,
        r: Array,
    ) -> Array:
        loss, grads = nnx.value_and_grad(loss_fn)(policy, s, a, r)
        opt.update(grads)
        return loss

    for _ in trange(n_iters, desc="Optimizing Policy"):
        states, actions, rewards, key = sample_trajectory(env, policy, key)

        s = jnp.array(states)
        a = jnp.array(actions)
        r = jnp.array(rewards)

        train_step(policy, opt, s, a, r)

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
