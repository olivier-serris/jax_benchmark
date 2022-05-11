from benchmark_utils import get_rollouts_times
import hydra
import logging
import os
import cProfile
from pstats import Stats
from omegaconf import OmegaConf, open_dict
import torch

# A logger for this file
log = logging.getLogger(__name__)


def single_time_experiment(cfg):
    """
    Launch an experiment with the env_name from brax
    show steps_per_seconds for a rollout worker.
    """
    setup_method = hydra.utils.get_method(cfg.method.path)
    rollout_fct = setup_method(cfg, cfg.n_pop, cfg.n_env, cfg.n_step)

    # warmup:
    get_rollouts_times(
        rollout_fct,
        cfg.n_pop,
        cfg.n_env,
        cfg.n_step,
        2,
        cfg.method.name,
    )
    if cfg.profiling:
        pr = cProfile.Profile()
        pr.enable()

    # real experiment:
    results = get_rollouts_times(
        rollout_fct,
        cfg.n_pop,
        cfg.n_env,
        cfg.n_step,
        cfg.n_repetition,
        cfg.method.name,
    )
    if cfg.profiling:
        pr.disable()
        stats = Stats(pr)
        stats = stats.sort_stats("tottime")
        stats.dump_stats(f"{os.getcwd()}/cProfile_stats.data")
        stats.print_stats(100)

    log.info(
        f"env={cfg.env_name},n_pop={cfg.n_pop}, n_env={cfg.n_env}, n_step={cfg.n_step}"
    )
    log.info("steps_per_seconds={:.1e}".format(results.step_per_sec))
    return results


@hydra.main(config_path=f"{os.getcwd()}/configs/", config_name="single_exp.yaml")
def main(cfg):
    # set the current device used :
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        current_device_id = torch.cuda.current_device()
        cfg.device = str(torch.cuda.get_device_name(current_device_id))
    cfg_save_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    single_time_experiment(cfg)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    main()


# how to visualize results with snakeviz :
# pip install snakeviz
# once installed, you can use snakeviz to view the file
# snakeviz /path/to/your/dump/pstat/file
