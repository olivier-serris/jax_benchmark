import jax
import chex
from brax import envs as brax_env
import brax.jumpy as jp
import jax.numpy as jnp


def get_action(rng, action_dim):
    # print("get_action is recompiled ")
    return jp.ones(action_dim)


def one_episode(env, n_step: int, rng_reset: chex.PRNGKey, rng_init: chex.PRNGKey):
    sim_state = env.reset(rng_reset)

    def policy_step(state_input: brax_env.State, tmp):
        """jp.scan compatible step transition in jax env."""
        state, rng = state_input
        rng, rng_action = jp.random_split(rng)
        action = get_action(rng_action, env.action_size)
        next_s = env.step(state, action)
        carry = [next_s, rng]
        return carry, [state.obs, state.reward, state.done, ]

    carry, scan_out = jp.scan(
        policy_step, [sim_state, rng_init], (), n_step
    )
    end_sim_state = carry[0]
    obs, reward, done,  = scan_out

    return (jnp.vstack([obs, end_sim_state.obs]),
            jnp.hstack([reward, end_sim_state.reward]),
            jnp.hstack([done, end_sim_state.done]),
            )


class mapP_mapE_laxS:
    '''
    Parallelism strategy only compatible with JAX models 
    Implement the pseudo code : 
    Vmap(pop):
        Vmap(env):
            lax(steps)
                action = np.ones()
                env.step(action)
    '''

    def __init__(self, env: brax_env.Env) -> None:
        self.env = env
        def action(rng): return get_action(rng, env.action_size)
        self.get_action = action

        env_episodes = jax.vmap(one_episode, in_axes=(None, None, 0, 0))

        all_episodes = jax.vmap(env_episodes, in_axes=(None, None, 0, 0))
        self.all_episodes = jax.jit(all_episodes, static_argnums=(0, 1))

    def rollout(self, rng: jax.random.KeyArray, n_pop: int, n_env: int, n_step: int):
        '''
        Execute complete episodes with n_pop individual, each on n_env environments.
        return a tuple [obs,rewards,dones]
            obs has shape [n_pop, n_env, n_step, obs_dim]
            rewards has shape [n_pop, n_env, n_step]
            dones has shape [n_pop, n_env, n_step]
        '''
        rng, *rng_reset = jax.random.split(rng, n_pop*n_env+1)
        rng_reset = jp.array(rng_reset).reshape(n_pop, n_env, -1)

        rng, *rng_init = jax.random.split(rng, n_pop*n_env+1)
        rng_init = jp.array(rng_init).reshape(n_pop, n_env, -1)

        results = self.all_episodes(self.env, n_step, rng_reset, rng_init)
        obs, reward, done = results
        return obs, reward, done


class forS_mapP_mapE:
    '''
    Parallelism strategy compatible with Non-JAX models 

    Implement the pseudo code : 
    for step in range(n_steps):
        action = np.ones(n_pop,n_env)
        Vmap(pop):
            Vmap(env):
                env.step(action)

    '''

    def __init__(self, env: brax_env.Env) -> None:
        self.env = env
        # vmap over mulitple pop
        pops_reset = jax.vmap(env.reset, in_axes=(0))
        # vmap over multiples env
        self.all_reset = jax.jit(jax.vmap(pops_reset, in_axes=(0)))

        # vmap over multiple envs
        pops_step = jax.vmap(env.step, in_axes=(0, 0))
        # vmap over multiple pops
        all_step = jax.vmap(pops_step, in_axes=(0, 0))
        self.all_step = jax.jit(all_step)

        # get actions :
        pops_actions = jax.vmap(get_action, in_axes=(0, None))
        all_actions = jax.vmap(pops_actions, in_axes=(0, None))
        self.all_actions = jax.jit(all_actions, static_argnums=1)

    def rollout(self, rng: jax.random.KeyArray, n_pop: int, n_env: int, n_step: int):
        '''
        Execute complete episodes with n_pop individual, each on n_env environments.
        return a tuple [obs, rewards, dones]
            obs has shape [n_pop, n_env,n_step, obs_dim]
            rewards has shape [n_pop, n_env, n_step]
            dones has shape [n_pop, n_env, n_step]
        '''

        rng, *rng_reset = jax.random.split(rng, n_pop*n_env+1)
        rng_reset = jp.array(rng_reset).reshape(n_pop, n_env, -1)

        sim_states = self.all_reset(rng_reset)
        results = [(sim_states.obs, sim_states.reward, sim_states.done)]
        for _ in range(n_step):
            rng, *rng_actions = jax.random.split(rng, n_pop*n_env+1)
            rng_actions = jp.array(rng_actions).reshape(n_pop, n_env, -1)
            actions = self.all_actions(rng_actions, self.env.action_size)
            sim_states = self.all_step(sim_states, actions)
            step_res = (sim_states.obs, sim_states.reward, sim_states.done)
            results.append(step_res)

        results = jax.tree_map(lambda *xs: jp.array(xs), *results)
        obs, reward, done = results
        obs = jnp.moveaxis(obs, (0, 1, 2, 3), (2, 0, 1, 3))
        reward = jnp.moveaxis(reward, (0, 1, 2), (2, 0, 1))
        done = jnp.moveaxis(done, (0, 1, 2), (2, 0, 1))
        return obs, reward, done
