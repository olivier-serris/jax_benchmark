import brax.envs as brax_env
import jax
from no_actors.rollout_no_actors import mapP_mapE_laxS, forS_mapP_mapE
from benchmark_grid_exp import get_rollouts_times


def verify_recompilations():
    """Check id jax compiles more often then expected."""
    rng = jax.random.PRNGKey(0)
    print("warm_up")

    env = brax_env.create(env_name="inverted_pendulum", legacy_spring=True)

    workers = {"fmm": forS_mapP_mapE(env), "mml": mapP_mapE_laxS(env)}

    all_kwargs = [
        {"n_pop": 2, "n_env": 2, "n_step": 2},
        {"n_pop": 3, "n_env": 2, "n_step": 2},
        {"n_pop": 2, "n_env": 3, "n_step": 2},
        {"n_pop": 2, "n_env": 2, "n_step": 3},
    ]
    for kwargs in all_kwargs:
        print("start_warmup ", kwargs)
        get_rollouts_times(workers, rng, **kwargs)
        print("end_warmup")
        print("\t same function", kwargs)
        # if we see recompilation here it's abnormal.
        get_rollouts_times(workers, rng, **kwargs)


if __name__ == "__main__":
    verify_recompilations()
