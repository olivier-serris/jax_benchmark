import brax.envs as brax_env
import jax

from jax_rollout.no_actors.exp_utils import with_jax_setup as no_actor_with_jax
from jax_rollout.no_actors.exp_utils import with_pytorch_setup as no_actor_with_pytorch
from jax_rollout.with_actors.exp_utils import with_pytorch_setup as actor_with_pytorch
from jax_rollout.with_actors.exp_utils import with_jax_setup as actor_with_jax
from jax_rollout.with_actors.rollout_pytorch import RolloutPytorch
from benchmark.benchmark_grid_exp import get_rollouts_times
import os
from hydra import compose, initialize


def verify_recompilations():
    """Check id jax compiles more often then expected."""

    # load a default config file  :
    initialize(config_path=f"./configs/", job_name="job_name")
    cfg = compose(config_name="mid_actor_100_000_step.yaml")

    print("warm_up")

    # env = brax_env.create(env_name="inverted_pendulum", legacy_spring=True)

    all_kwargs = [
        {"n_pop": 2, "n_env": 2, "n_step": 2},
        {"n_pop": 3, "n_env": 2, "n_step": 2},
        {"n_pop": 2, "n_env": 3, "n_step": 2},
        {"n_pop": 2, "n_env": 2, "n_step": 3},
    ]

    rollout_contructors = {
        "fmm": no_actor_with_pytorch,
        "mms": no_actor_with_jax,
        "fmm_torch": actor_with_pytorch,
        "mms_jax": actor_with_jax,
    }
    for label, rollout_constructor in rollout_contructors.items():
        print("_" * 10, label, "_" * 10)
        for kwargs in all_kwargs:
            rollout_fct = rollout_constructor(cfg, **kwargs)

            print("start_warmup ", kwargs)
            get_rollouts_times(rollout_fct, n_repetition=2, method_name=label, **kwargs)
            print("end_warmup")
            print("same function", kwargs)
            # if we see recompilation here it's abnormal.
            get_rollouts_times(rollout_fct, n_repetition=2, method_name=label, **kwargs)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    verify_recompilations()
