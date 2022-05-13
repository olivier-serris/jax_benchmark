import torch
import hydra
import hydra
import time
import pandas as pd
import os
import torch
from plots.plot_results import multi_plot_jax_vs_pytorch
from benchmark.benchmark_utils import time_experiments
from omegaconf import OmegaConf, open_dict

# have torch allocate on device first, to prevent JAX from swallowing up all the
# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
v = torch.ones(1, device="cuda")


# to change with command line : --config-name new_config
@hydra.main(
    config_path=f"{os.getcwd()}/configs/", config_name="mid_actor_100_000_step.yaml"
)
def main(cfg):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # set the current device used :
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        current_device_id = torch.cuda.current_device()
        cfg.device = str(torch.cuda.get_device_name(current_device_id))
    cfg_save_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
    OmegaConf.save(cfg, cfg_save_path)

    start = time.time()
    results = time_experiments(cfg)
    df = pd.DataFrame(results)
    df.to_csv(cfg.save_filename + ".csv")
    print("total exp time: ", time.time() - start)
    # todo verify multiplot location :
    multi_plot_jax_vs_pytorch(
        df,
        cfg.env_name,
        cfg.device,
        f"{os.getcwd()}/{cfg.save_filename}.png",
    )


if __name__ == "__main__":
    main()
