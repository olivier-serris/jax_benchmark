import jax
import chex
from brax import envs as brax_env
import functools
import gym
from typing import Any, Callable, List, Optional, Union
from brax.envs import wrappers, to_torch
import torch
import torch.nn as nn
from jax_rollout.with_actors.rollout_data import RolloutData
from dataclasses import dataclass
import dataclasses
from functorch import combine_state_for_ensemble
import functorch

# TODO :
# add generic way to specify keys
# from the info dict to gather
@dataclass
class StepData:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor


# TODO check efficiency of * operator.


def dataclass_map(fct: Callable[[List], Any], *elements: dataclass) -> dataclass:
    """ """
    fields = dataclasses.fields(elements[0])
    new_vals = {}
    for field in fields:
        new_vals[field.name] = fct([getattr(el, field.name) for el in elements])
    DataType = type(elements[0])
    return DataType(**new_vals)


# def tuple_map(fct: Callable[[List], Any], elements: List[Tuple]) -> dataclass:
#     """ """
#     tuple_len = len(elements[0])
#     new_vals = []
#     for i in range(tuple_len):
#         mapped_value = fct([el[i] for el in elements])
#         new_vals.append(mapped_value)
#     return Tuple(*new_vals)


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    n_pop: int = 1,
    n_env: int = 1,
    eval_metrics: bool = False,
    **kwargs,
) -> brax_env.Env:
    """Creates an Env with a specified brax system."""
    # TODO : Check optimal order (performance-wise) of wrappers :
    env = brax_env._envs[env_name](**kwargs)
    if episode_length is not None:
        env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)

    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    # if n_env > 1:
    env = wrappers.VectorWrapper(env, n_env)
    # if n_pop > 1:
    env = wrappers.VectorWrapper(env, n_pop)
    if eval_metrics:
        env = wrappers.EvalWrapper(env)

    return env


def create_gym_env(
    env_name: str,
    n_env: Optional[int] = None,
    n_pop: Optional[int] = None,
    seed: int = 0,
    backend: Optional[str] = None,
    **kwargs,
) -> Union[gym.Env, gym.vector.VectorEnv]:
    """Convert Brax env to gym env"""
    environment = create(env_name=env_name, n_env=n_env, n_pop=n_pop, **kwargs)
    if n_env <= 0:
        raise ValueError("`batch_size` should either be None or a positive integer.")
    if n_pop <= 0:
        raise ValueError("`batch_size` should either be None or a positive integer.")
    if (n_env is None or n_env == 1) and (n_pop is None or n_pop == 1):
        return wrappers.GymWrapper(environment, seed=seed, backend=backend)

    return wrappers.VectorGymWrapper(environment, seed=seed, backend=backend)


class RolloutPytorch:
    """
    Parallelism strategy compatible with Non-JAX models

    Implement the pseudo code :
    for step in range(n_steps):
        action = np.ones(n_pop,n_env)
        obs = Vmap(pop) -> Vmap(env)  -> env.step(action)
    """

    def __init__(
        self,
        env_name: brax_env.Env,
        n_pop: int,
        n_env: int,
        n_step: int,
        device="cuda",
    ) -> None:
        self.n_pop = n_pop
        self.n_env = n_env
        self.n_step = n_step

        gym_name = "brax-" + env_name + "-v0"
        if gym_name in gym.envs.registry.env_specs:
            del gym.envs.registry.env_specs[gym_name]
        entry_point = functools.partial(create_gym_env, env_name=env_name)
        gym.register(gym_name, entry_point=entry_point)
        env = gym.make(gym_name, n_pop=n_pop, n_env=n_env, episode_length=n_step)
        self.env = to_torch.JaxToTorchWrapper(env, device=device)

    def set_actors(self, actors: List[nn.Module]):
        predict, self.params, self.buffers = combine_state_for_ensemble(actors)
        # TODO: buffer are ignored, see i
        self.all_actions = functorch.vmap(predict)

    def reset(self) -> None:
        self.last_obs = self.env.reset()

    def rollout(self, reset: bool = False) -> chex.Array:
        """
        Execute complete episodes with n_pop individual, each on n_env environments.
        return a tuple [obs, rewards, dones]
            obs has shape [n_pop, n_env,n_step, obs_dim]
            rewards has shape [n_pop, n_env, n_step]
            dones has shape [n_pop, n_env, n_step]
        """
        if reset:
            self.env.reset()

        cur_obs = self.last_obs

        results = []
        for _ in range(self.n_step):
            with torch.no_grad():
                actions = self.all_actions(
                    self.params, self.buffers, cur_obs.view(self.n_pop, self.n_env, -1)
                )
            next_obs, reward, done, info = self.env.step(actions)
            truncation = (
                info["Timelimit.truncated"]
                if "Timelimit.truncated" in info
                else torch.zeros_like(done, dtype=bool)
            )
            step_res = StepData(cur_obs, actions, reward, done, truncation)
            results.append(step_res)
            cur_obs = next_obs
        self.last_obs = cur_obs

        def gather_results(*results):
            res = torch.stack(*results)
            res = res.reshape((self.n_step, self.n_pop, self.n_env, -1))
            res = torch.moveaxis(res, (0, 1, 2), (2, 0, 1))
            return res

        all_steps = dataclass_map(gather_results, *results)

        return RolloutData(
            sim_states=None,
            obs=all_steps.obs,
            rewards=all_steps.reward,
            dones=all_steps.done,
            truncation=all_steps.truncated,
        )
