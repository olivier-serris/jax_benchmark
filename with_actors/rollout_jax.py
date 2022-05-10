from typing import Callable, Optional
import jax
import chex
from brax import envs as brax_env
import brax.jumpy as jp
import jax.numpy as jnp

from .rollout_data import RolloutData


class RolloutJax:
    """
    Parallelism strategy only compatible with JAX models
    Implement the pseudo code :
    Vmap(pop):
        Vmap(env):
            lax(steps)
                action = np.ones()
                env.step(action)
    """

    def __init__(
        self,
        network_call: Callable,
        env: brax_env.Env,
        n_pop: int,
        n_env: int,
        n_step: int,
    ) -> None:
        self.env = env
        self.n_pop = n_pop
        self.n_env = n_env
        self.n_step = n_step
        self.network_call = network_call

        env_episodes = jax.vmap(self.collect_n_step, in_axes=(0, 0, None))
        all_episodes = jax.vmap(env_episodes, in_axes=(0, 0, 0))
        self.all_episodes = jax.jit(all_episodes)

        reset_envs = jax.vmap(self.env.reset, in_axes=(0,))
        all_reset = jax.vmap(reset_envs, in_axes=(0,))
        self.all_reset = jax.jit(all_reset)

    def reset(self, rng: jax.random.KeyArray):
        """
        Class reset in all environement,
        returns a starting state for each of the selected environments.

        input :
        @rng of shape 1
        returns :
        simulator states of shape [n_pop][n_env]
        """
        rng_reset = jax.random.split(rng, self.n_pop * self.n_env).reshape(
            self.n_pop, self.n_env, -1
        )
        sim_states = self.all_reset(rng_reset)
        return sim_states

    def rollout(
        self,
        policy_params: chex.ArrayTree,
        rng: jax.random.KeyArray,
        sim_states: Optional[brax_env.State] = None,
    ):
        """
        Execute complete episodes with n_pop individual, each on n_env environments.

        return a tuple [obs,rewards,dones]
            obs has shape [n_pop, n_env, n_step, obs_dim]
            rewards has shape [n_pop, n_env, n_step]
            dones has shape [n_pop, n_env, n_step]
        """
        if sim_states is None:
            rng, reset_key = jax.random.split(rng)
            sim_states = self.reset(reset_key)

        rng_init = jax.random.split(rng, self.n_pop * self.n_env)
        rng_init = rng_init.reshape(self.n_pop, self.n_env, -1)

        results = self.all_episodes(rng_init, sim_states, policy_params)
        state, obs, reward, done, truncation = results
        # TODO : check of to get truncation with brax
        return RolloutData(
            sim_states=state, obs=obs, rewards=reward, dones=done, truncation=truncation
        )

    def collect_n_step(
        self,
        rng: jax.random.KeyArray,
        sim_state: brax_env.State,
        policy_params: chex.ArrayTree,
    ):
        def collect_one_step(state_input: brax_env.State, tmp):
            """jp.scan compatible step transition in jax env."""
            # print("collect_one_step is compiled")
            state, rng = state_input
            rng, rng_action = jp.random_split(rng)

            action = self.network_call(
                {"params": policy_params}, state.obs, rng=rng_action
            )
            next_s = self.env.step(state, action)
            carry = [next_s, rng]
            return carry, [
                state.obs,
                state.reward,
                state.done,
                state.info["truncation"],
            ]

        carry, scan_out = jp.scan(collect_one_step, [sim_state, rng], (), self.n_step)
        end_sim_state = carry[0]
        obs, reward, done, truncation = scan_out

        return (
            end_sim_state,
            jnp.vstack([obs, end_sim_state.obs]),
            jnp.hstack([reward, end_sim_state.reward]),
            jnp.hstack([done, end_sim_state.done]),
            jnp.hstack([truncation, end_sim_state.info["truncation"]]),
        )
