import hydra
from tqdm import tqdm
import time
import numpy as np
import brax.envs as brax_env
import jax
import pandas as pd
from dataclasses import dataclass
from typing import Dict
import os
from typing import List
from omegaconf import OmegaConf, open_dict
import torch


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
    workers: Dict, rng: jax.random.KeyArray, n_pop: int, n_env: int, n_step: int
) -> List[DataPoint]:
    results = []
    total_steps = np.prod([n_pop, n_env, n_step])
    for label, cls in list(workers.items()):
        start = time.time()
        rollout_data = cls.rollout(rng, n_pop, n_env, n_step)
        for data in rollout_data:
            data.block_until_ready()
        total_time = time.time() - start
        exp_data = DataPoint(
            n_step,
            n_env,
            n_pop,
            total_time=total_time,
            method=label,
            total_steps=total_steps,
            step_per_sec=total_steps / total_time,
        )
        results.append(exp_data)

    return results


def time_experiments(cfg) -> List[DataPoint]:

    # set the current device used :
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        current_device_id = torch.cuda.current_device()
        cfg.device = str(torch.cuda.get_device_name(current_device_id))
    cfg_save_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
    OmegaConf.save(cfg, cfg_save_path)

    # Creates Env :
    env = brax_env.create(env_name=cfg.env_name, legacy_spring=True)
    rng = jax.random.PRNGKey(0)

    # all values :
    print("launch exp :")
    data_points = []
    for parameters in tqdm(list(cfg.configurations), disable=False):
        n_pop, n_env = parameters["n_pop"], parameters["n_env"]
        n_step = parameters["n_step"]

        workers = {}
        for el in cfg.classes.items():
            label, cls = el
            workers[label] = hydra.utils.instantiate(cls, env=env)

        # warm_up :
        get_rollouts_times(workers, rng, n_pop, n_env, n_step)
        # real experiment:
        for _ in tqdm(list(range(cfg.n_repetition))):
            results = get_rollouts_times(workers, rng, n_pop, n_env, n_step)
            data_points += results
    return data_points


# @hydra.main(
#     config_path=f"{os.getcwd()}/configs/", config_name="no_actor_10_000_step.yaml"
# )
@hydra.main(
    config_path=f"{os.getcwd()}/configs/", config_name="no_actor_100_000_step.yaml"
)
def main(cfg):
    start = time.time()
    results = time_experiments(cfg)
    df = pd.DataFrame(results)
    df.to_csv(cfg.save_filename)
    print("total exp time: ", time.time() - start)


if __name__ == "__main__":
    main()
