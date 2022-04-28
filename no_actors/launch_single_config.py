from launch_grid_exp import get_rollouts_times
from tqdm import tqdm
import jax
import brax.envs as brax_env
import argparse
from rollout_no_actors import mapP_mapE_laxS, forS_mapP_mapE
import numpy as np


def single_time_experiment(env_name, n_pop, n_env, n_step, n_repetition):
    '''
        Launch an experiment with the env_name from brax 
        show steps_per_seconds for a rollout worker.
    '''
    env = brax_env.create(env_name=env_name,
                          legacy_spring=True)
    rng = jax.random.PRNGKey(0)
    workers = {
        'mml': mapP_mapE_laxS(env),
        'fmm': forS_mapP_mapE(env)
    }
    # warmup:
    get_rollouts_times(workers, rng, n_pop, n_env, n_step)
    # real experiment:
    data_points = []
    for _ in tqdm(list(range(n_repetition))):
        results = get_rollouts_times(workers,
                                     rng, n_pop, n_env, n_step)
        data_points += results
    step_per_seconds = np.mean([data.step_per_sec for data in data_points])
    print(f'env={env_name},n_pop={n_pop}, n_env={n_env}, n_step={n_step}')
    print('steps_per_seconds=', '{:.1e}'.format(step_per_seconds))
    return data_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str,
                        default='inverted_pendulum', help='brax environment name')
    parser.add_argument('--n_repetition', type=int,
                        default=1, help='number of repetitation for the experiment')
    parser.add_argument('--n_pop', type=int,
                        default=3, help='size of population')
    parser.add_argument('--n_env', type=int,
                        default=2, help='number of environments for each ind')
    parser.add_argument('--n_step', type=int,
                        default=16, help='number step for each episodes')
    args = parser.parse_args()

    single_time_experiment(env_name=args.env_name,
                           n_pop=args.n_pop,
                           n_env=args.n_env,
                           n_step=args.n_step,
                           n_repetition=args.n_repetition)


if __name__ == '__main__':
    main()
