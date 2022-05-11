from brax import envs as brax_env
import chex
import jax.numpy as jnp

# from IPython.display import HTML, Image


@chex.dataclass
class RolloutData:
    sim_states: brax_env.State
    obs: chex.Array
    rewards: chex.Array
    dones: chex.Array
    truncation: chex.Array

    @staticmethod
    def to_pop_fitness(rollout: "RolloutData") -> chex.Array:
        """Returns the mean fitness of each individual over multiple episodes.
        fitness shape = [npop][n_env]
        """
        # TODO : be able to compute fitness of a sequence of auto-reset episodes
        rewards, dones = rollout.rewards, rollout.dones
        crewards = jnp.cumsum(rewards, axis=-1)  # cumulated rewards
        mask_first_done = jnp.logical_and(dones == 1, jnp.cumsum(dones, axis=-1) == 1)

        pop_fitness = jnp.sum(crewards * mask_first_done, axis=(-1))
        return pop_fitness

    @staticmethod
    def to_transition(rollout: "RolloutData") -> "RolloutData":
        pass

    def to_video():
        pass
        # HTML(html.render(env.sys, [s.qp for s in rollout]))
        # Image(image.render(env.sys, [s.qp for s in rollout], width=320, height=240))
