import pytest
from jax import make_jaxpr

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax, GAILJax, AMPJax
from loco_mujoco.utils import MetricsHandler

from test_conf import *


# Set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


def test_PPO_Jax_build_train_fn(ppo_rl_config):

    config = ppo_rl_config

    # get task factory
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # create env
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # get initial agent configuration
    agent_conf = PPOJax.init_agent_conf(env, config)

    # build training function
    train_fn = PPOJax.build_train_fn(env, agent_conf)

    # jit and vmap training function
    train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    # Use make_jaxpr to check if the function compiles correctly
    try:
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))

        jaxpr = make_jaxpr(train_fn)(_rng)

        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")


@pytest.mark.parametrize("algorithm", ("GAIL", "AMP"))
def test_Imitation_Jax_build_train_fn(algorithm, imitation_config):

    alg_cls = GAILJax if algorithm == "GAIL" else AMPJax

    config = imitation_config

    # get task factory
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

    # create env
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # create an expert dataset
    expert_dataset = env.create_dataset()

    # get initial agent configuration
    agent_conf = alg_cls.init_agent_conf(env, config)
    agent_conf = agent_conf.add_expert_dataset(expert_dataset)

    # setup metric handler (optional)
    mh = MetricsHandler(config, env) if config.experiment.validation.active else None

    # build training function
    train_fn = alg_cls.build_train_fn(env, agent_conf, mh=mh)

    # jit and vmap training function
    train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

    # Use make_jaxpr to check if the function compiles correctly
    try:
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))

        jaxpr = make_jaxpr(train_fn)(_rng)

        assert jaxpr is not None
    except Exception as e:
        pytest.fail(f"JAX function compilation failed: {e}")
