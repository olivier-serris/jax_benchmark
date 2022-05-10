import torch
import hydra
import hydra
from tqdm import tqdm
import time
import numpy as np
import jax
import pandas as pd
import os
from typing import List
from omegaconf import OmegaConf, open_dict
import torch
import timeit
from dataclasses import dataclass
import gc
from plots.plot_results import multi_plot

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device="cuda")


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

    # set the current device used :
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        current_device_id = torch.cuda.current_device()
        cfg.device = str(torch.cuda.get_device_name(current_device_id))
    cfg_save_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
    OmegaConf.save(cfg, cfg_save_path)

    # all values :
    print("launch exp :")
    data_points = []
    for parameters in tqdm(list(cfg.configurations), disable=False):
        n_pop, n_env = parameters["n_pop"], parameters["n_env"]
        n_step = parameters["n_step"]

        for el in cfg.classes.items():
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
            gc.collect()
            data_points.append(result)
    return data_points


# TODO : put torch tensor into memory for hybrid methods.
@hydra.main(
    config_path=f"{os.getcwd()}/configs/", config_name="no_actor_100_000_step.yaml"
)
def main(cfg):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    start = time.time()
    results = time_experiments(cfg)
    df = pd.DataFrame(results)
    df.to_csv(cfg.save_filename + ".csv")
    print("total exp time: ", time.time() - start)
    # todo veriry multiplot location :
    multi_plot(
        df,
        cfg.env_name,
        cfg.device,
        f"{os.getcwd()}/{cfg.save_filename}.png",
    )


if __name__ == "__main__":
    main()
