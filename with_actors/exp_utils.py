from typing import List
import chex
from numpy import isin
from with_actors.rollout_pytorch import RolloutPytorch
from with_actors.rollout_jax import RolloutJax
import torch
import hydra
import hydra
import brax.envs as brax_env
import jax
import torch
import jax.numpy as jnp
from dataclasses import fields
import torch.nn as nn


def with_pytorch_setup(cfg, n_pop, n_env, n_step):
    """
    Returns a rollout function compatible with timeit.timeit()
    as it has captured the necessary context.
    This rollout function is a test for rollouts with pytorch networks.
    """

    assert torch.cuda.is_available()

    rollout_worker = RolloutPytorch(
        cfg.env_name,
        n_pop=n_pop,
        n_env=n_env,
        n_step=n_step,
        device="cuda",
    )

    actors: List[nn.Module] = [
        hydra.utils.instantiate(
            cfg.pytorch_model,
            in_dim=rollout_worker.env.observation_space.shape[-1],
            out_dim=rollout_worker.env.action_space.shape[-1],
        )
        for _ in range(n_pop)
    ]
    for actor in actors:
        actor.to("cuda")

    rng = jax.random.PRNGKey(0)
    rollout_worker.reset()

    # dict captured by execute_rollout to carry values between exections.
    carry = {
        "rollout_cls": rollout_worker,
        "actors": actors,
        "seed": rng,
    }

    def execute_rollout():
        rng = carry["seed"]
        """launch a rollout and make sure that the execution is finished."""
        rollout_worker: RolloutPytorch = carry["rollout_cls"]
        actors = carry["actors"]
        rollout_data = rollout_worker.rollout(actors, rng, reset=False)

        for field in fields(rollout_data):
            data = getattr(rollout_data, field.name)
            if isinstance(data, chex.Array):
                data.block_until_ready()

        carry["seed"] = jax.random.split(carry["seed"], 2)

    return execute_rollout


def with_jax_setup(cfg, n_pop, n_env, n_step):
    """
    Returns a rollout function compatible with timeit.timeit()
    as it has captured the necessary context.
    This rollout function is used for performance benchmark rollouts with jax networks.
    """

    env = brax_env.create(env_name=cfg.env_name, legacy_spring=cfg.legacy_spring)
    network = hydra.utils.instantiate(cfg.jax_model, out_dim=env.action_size)

    rng = jax.random.PRNGKey(0)
    rng, init_key, reset_key = jax.random.split(rng, 3)

    dummy_obs = jnp.ones((10, env.observation_size))

    all_init = jax.vmap(network.init, (None, None, 0))
    pop_params = all_init(rng, dummy_obs, jax.random.split(init_key, n_pop))
    pop_params = pop_params["params"]

    rollout_worker = RolloutJax(network.apply, env, n_pop, n_env, n_step)

    # dict captured by execute_rollout to carry values between exections.
    carry = {
        "rollout_obj": rollout_worker,
        "seed": rng,
        "sim_states": rollout_worker.reset(reset_key),
    }

    def execute_rollout():
        carry["seed"], key = jax.random.split(carry["seed"], 2)
        rollout_worker: RolloutJax = carry["rollout_obj"]
        sim_states = carry["sim_states"]
        rollout_data = rollout_worker.rollout(pop_params, key, sim_states=sim_states)

        for field in fields(rollout_data):
            data = getattr(rollout_data, field.name)
            if isinstance(data, chex.Array):
                data.block_until_ready()

        carry["sim_states"] = rollout_data.sim_states

    return execute_rollout
