import torch
import hydra
import hydra
import time
import pandas as pd
import os
import torch
from dataclasses import dataclass
from plots.plot_results import multi_plot
from benchmark_utils import time_experiments

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device="cuda")


# to change with command line : --config-name new_config
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
    # todo verify multiplot location :
    multi_plot(
        df,
        cfg.env_name,
        cfg.device,
        f"{os.getcwd()}/{cfg.save_filename}.png",
    )


if __name__ == "__main__":
    main()
