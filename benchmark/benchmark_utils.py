import torch
import hydra
import hydra
from tqdm import tqdm
import numpy as np
import os
from typing import List
from omegaconf import OmegaConf, open_dict
import torch
import timeit
from dataclasses import dataclass
import gc


@dataclass
class DataPoint:
    n_step: int
    n_env: int
    n_pop: int
    total_time: float
    total_steps: float
    step_per_sec: float
    method: str


def get_rollouts_times(
    execute_rollout,
    n_pop: int,
    n_env: int,
    n_step: int,
    n_repetition: int,
    method_name: str,
) -> List[DataPoint]:
    total_steps = np.prod([n_pop, n_env, n_step])
    elapsed = timeit.Timer(execute_rollout).timeit(n_repetition)
    mean_time = elapsed / n_repetition
    exp_data = DataPoint(
        n_step,
        n_env,
        n_pop,
        total_time=mean_time,
        method=method_name,
        total_steps=total_steps,
        step_per_sec=total_steps / mean_time,
    )
    return exp_data


def time_experiments(cfg) -> List[DataPoint]:
    # all values :
    print("launch exp :")
    data_points = []
    for parameters in tqdm(list(cfg.configurations), disable=False):
        n_pop, n_env = parameters["n_pop"], parameters["n_env"]
        n_step = parameters["n_step"]

        for el in cfg.classes.items():
            gc.collect()
            label, setup_path = el
            construct_rollout_fct = hydra.utils.get_method(setup_path)
            rollout_fct = construct_rollout_fct(cfg, n_pop, n_env, n_step)
            # warm_up :
            get_rollouts_times(rollout_fct, n_pop, n_env, n_step, 2, label)
            # real experiment:
            result = get_rollouts_times(
                rollout_fct,
                n_pop,
                n_env,
                n_step,
                n_repetition=cfg.n_repetition,
                method_name=label,
            )
            data_points.append(result)
    return data_points
