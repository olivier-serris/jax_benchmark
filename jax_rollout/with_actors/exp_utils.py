from typing import List
import chex
from jax_rollout.with_actors.rollout_pytorch import RolloutPytorch
from jax_rollout.with_actors.rollout_jax import RolloutJax
import torch
import hydra
import hydra
import brax.envs as brax_env
import jax
import torch
import jax.numpy as jnp
from dataclasses import fields
import torch.nn as nn
import gym
import envpool
import functorch
from functorch import combine_state_for_ensemble


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
        device=cfg.torch_device,
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
        actor.to(cfg.torch_device)

    rollout_worker.set_actors(actors)
    rollout_worker.reset()

    # dict captured by execute_rollout to carry values between exections.
    carry = {
        "rollout_cls": rollout_worker,
    }

    def execute_rollout():
        """launch a rollout and make sure that the execution is finished."""
        rollout_worker: RolloutPytorch = carry["rollout_cls"]
        rollout_data = rollout_worker.rollout(reset=False)

        for field in fields(rollout_data):
            data = getattr(rollout_data, field.name)
            if isinstance(data, chex.Array):
                data.block_until_ready()

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


def with_gym_setup(cfg, n_pop, n_env, n_step):
    env_name = str.capitalize(cfg.env_name) + "-" + cfg.gym_version
    env = gym.vector.make(env_name, num_envs=n_env)

    actors: List[nn.Module] = [
        hydra.utils.instantiate(
            cfg.pytorch_model,
            in_dim=env.observation_space.shape[-1],
            out_dim=env.action_space.shape[-1],
        )
        for _ in range(n_pop)
    ]
    for actor in actors:
        actor.to(cfg.torch_device)

    carry = {
        "obs": torch.tensor(env.reset(), device=cfg.torch_device, dtype=torch.float),
    }

    def execute_rollout():
        obs = carry["obs"]
        with torch.no_grad():
            for actor in actors:
                for _ in range(n_step):
                    action = actor(obs).cpu().numpy()
                    obs, rewards, dones, infos = env.step(action)
                    obs = torch.tensor(obs, device=cfg.torch_device, dtype=torch.float)

        carry["obs"] = obs

    return execute_rollout


def with_envpool_sync_setup(cfg, n_pop, n_env, n_step):
    import multiprocessing

    num_cpu = multiprocessing.cpu_count()
    env_name = str.capitalize(cfg.env_name) + "-" + cfg.gym_version
    num_threads = min(num_cpu, n_env * n_step)
    print("num_theads : ", num_threads)
    env = envpool.make(
        env_name, env_type="gym", num_envs=n_env * n_pop, num_threads=num_threads
    )

    actors: List[nn.Module] = [
        hydra.utils.instantiate(
            cfg.pytorch_model,
            in_dim=env.observation_space.shape[-1],
            out_dim=env.action_space.shape[-1],
        )
        for _ in range(n_pop)
    ]
    for actor in actors:
        actor.to(cfg.torch_device)

    predict, params, buffers = combine_state_for_ensemble(actors)
    all_actions = functorch.vmap(predict)

    carry = {
        "obs": torch.tensor(env.reset(), device=cfg.torch_device, dtype=torch.float),
        "params": params,
        "buffers": buffers,
    }

    def execute_rollout():
        obs = carry["obs"]
        params, buffers = carry["params"], carry["buffers"]
        with torch.no_grad():
            for _ in range(n_step):
                actions = all_actions(params, buffers, obs.view(n_pop, n_env, -1))
                actions = actions.reshape(n_pop * n_env, -1).cpu().numpy()
                obs, rewards, dones, infos = env.step(actions)
                obs = torch.tensor(obs, device=cfg.torch_device, dtype=torch.float)

        carry["obs"] = obs

    return execute_rollout
