from jax_rollout.no_actors.rollout_no_actors import forS_mapP_mapE, mapP_mapE_laxS
import brax.envs as brax_env
import jax
import gym
import numpy as np


def with_pytorch_setup(cfg, n_pop, n_env, n_step):
    """
    Returns a rollout function compatible with timeit.timeit()
    as it has captured the necessary context.
    This rollout function is a test for pytorch compatible rollouts
    """

    env = brax_env.create(env_name=cfg.env_name, legacy_spring=cfg.legacy_spring)
    rollout_worker = forS_mapP_mapE(env)

    carry = {
        "seed": jax.random.PRNGKey(0),
    }

    def execute_rollout():
        rng = carry["seed"]
        rollout_data = rollout_worker.rollout(rng, n_pop, n_env, n_step)
        for data in rollout_data:
            data.block_until_ready()
        carry["seed"] = jax.random.split(carry["seed"], 2)

    return execute_rollout


def with_jax_setup(cfg, n_pop, n_env, n_step):
    """
    Returns a rollout function compatible with timeit.timeit()
    as it has captured the necessary context.
    This rollout function is a test for rollouts with jax networks.
    """

    env = brax_env.create(env_name=cfg.env_name, legacy_spring=cfg.legacy_spring)
    rollout_worker = mapP_mapE_laxS(env)

    carry = {
        "seed": jax.random.PRNGKey(0),
    }

    def execute_rollout():
        rng = carry["seed"]
        """launch a rollout and make sure that the execution is finished."""
        rollout_data = rollout_worker.rollout(rng, n_pop, n_env, n_step)
        for data in rollout_data:
            data.block_until_ready()
        carry["seed"] = jax.random.split(carry["seed"], 2)

    return execute_rollout


def with_gym_setup(cfg, n_pop, n_env, n_step):
    env_name = str.capitalize(cfg.env_name) + "-" + cfg.gym_version
    env = gym.vector.make(env_name, num_envs=n_env)

    def execute_rollout():
        for _ in range(n_pop):
            for _ in range(n_step):
                action = np.ones(env.action_space.shape[-1])
                obs, rewards, dones, infos = env.step(action)

    return execute_rollout
